import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from cellpose import models, io
import os
import logging
from tqdm import tqdm
from datetime import datetime
import sys
import time

logger = logging.getLogger(__name__)


def segment_nuclei_cellpose_direct(
    h5_path,
    output_h5_path=None,
    gpu=False,
    visualize=False,
    model_type="nuclei",
    diameter=None,
    max_cells=None,
    plane_selection="middle",
    overwrite=False,
    cells_per_batch=10,
):
    """
    Segmentiert Nukleai direkt aus H5-Datei und fügt nuclei_seg Kanal hinzu.
    Verarbeitet Zellen in Batches (sammelt mehrere Zellen, dann segmentiert alle Planes).

    WICHTIG: Cellpose v4 verarbeitet 2D-Bilder in Listen IMMER sequentiell.
    Der batch_size-Parameter (Standard:64) kontrolliert GPU-Parallelisierung INNERHALB
    jedes Bildes durch Verarbeitung mehrerer 256x256-Patches gleichzeitig.

    cells_per_batch sollte hoch sein (10-20) um Overhead zu reduzieren.

    Parameter:
    - h5_path: Pfad zur Input H5-Datei
    - output_h5_path: Pfad zur Output H5-Datei (None = bearbeitet Input-Datei direkt)
    - gpu: GPU verwenden
    - visualize: Ergebnisse visualisieren
    - model_type: Cellpose-Modelltyp ('nuclei', 'cyto', 'cyto2')
    - diameter: Nukleaidurchmesser (None = automatisch)
    - max_cells: Maximale Anzahl zu verarbeitender Zellen
    - plane_selection: Welche Planes verarbeiten ('middle', 'first', 'last', 'all')
    - overwrite: Bestehende nuclei_seg überschreiben
    - cells_per_batch: Anzahl Zellen pro Batch (Standard: 10, höher = effizienter)
    """

    # Bestimme Output-Pfad - IMMER neue Datei in /myhome/iris erstellen
    if output_h5_path is None:
        # Erstelle Output in /myhome/iris (wo Speicherplatz verfügbar ist)
        input_basename = os.path.splitext(os.path.basename(h5_path))[0]
        output_h5_path = f"/myhome/iris/data/{input_basename}_with_nuclei_seg.h5"
        logger.info(f"Erstelle neue Datei mit nuclei_seg in: {output_h5_path}")

    mode = "w"  # Immer neue Datei erstellen

    # Überprüfe verfügbaren Speicherplatz
    import shutil

    output_dir = os.path.dirname(output_h5_path)
    free_bytes = shutil.disk_usage(output_dir).free
    free_gb = free_bytes / (1024**3)

    if free_gb < 1.0:
        raise RuntimeError(
            f"Nicht genügend Speicherplatz in {output_dir}. Verfügbar: {free_gb:.2f} GB"
        )

    logger.info(f"Verfügbarer Speicherplatz in {output_dir}: {free_gb:.2f} GB")

    # Cellpose Modell laden
    logger.info(f"Lade Cellpose-Modell: {model_type}, GPU: {gpu}")
    model = models.CellposeModel(gpu=gpu, model_type=model_type)

    # Metadaten für Segmentierung
    segmentation_metadata = {
        "cellpose_model": model_type,
        "diameter": diameter if diameter else "auto",
        "segmentation_date": datetime.now().isoformat(),
        "plane_selection": plane_selection,
        "source_file": os.path.basename(h5_path),
    }

    # Öffne neue H5-Datei und kopiere + erweitere Daten
    # Using context managers ensures files are properly closed even on exceptions
    try:
        with h5py.File(output_h5_path, mode) as output_file:

            # Kopiere alle Originaldaten aus Input-Datei
            logger.info("Kopiere ursprüngliche Datenstruktur...")
            with h5py.File(h5_path, "r") as input_file:

                # Füge Metadaten zur Output-Datei hinzu
                output_file.attrs["source_file"] = os.path.basename(h5_path)
                output_file.attrs["creation_date"] = datetime.now().isoformat()
                output_file.attrs["purpose"] = "complete_with_nuclei_segmentation"

                cell_names = list(input_file.keys())
                total_cells = (
                    min(len(cell_names), max_cells) if max_cells else len(cell_names)
                )

                logger.info(
                    f"Verarbeite {total_cells} Zellen aus {len(cell_names)} verfügbaren"
                )

                # Verarbeite jede Zelle - sammle zuerst alle Daten für Batch-Processing
                processed_count = 0
                failed_count = 0
                total_nuclei = 0

                # Verarbeite Zellen in Batches (z.B. 2 Zellen gleichzeitig)
                cell_batch_data = []  # Sammelt Daten für mehrere Zellen

                for cell_idx, cell_name in enumerate(
                    tqdm(
                        cell_names[:total_cells],
                        desc="Kopiere & Segmentiere Zellen",
                    )
                ):
                    try:
                        # Kopiere ursprüngliche Zell-Daten
                        input_file.copy(cell_name, output_file)
                        cell_group = output_file[cell_name]

                        if "405" not in cell_group:
                            logger.warning(f"Kein 405-Kanal in Zelle {cell_name}")
                            failed_count += 1
                            continue

                        # Prüfe ob nuclei_seg bereits existiert
                        if "nuclei_seg" in cell_group and not overwrite:
                            logger.info(
                                f"nuclei_seg bereits vorhanden für {cell_name}, überspringe"
                            )
                            processed_count += 1
                            continue

                        # Erstelle/überschreibe nuclei_seg Gruppe
                        if "nuclei_seg" in cell_group:
                            del cell_group["nuclei_seg"]
                        nuclei_seg_group = cell_group.create_group("nuclei_seg")

                        # Füge Metadaten hinzu
                        for key, value in segmentation_metadata.items():
                            nuclei_seg_group.attrs[key] = value

                        # Lade 405-Kanal Daten
                        channel_405 = cell_group["405"]
                        plane_names = list(channel_405.keys())

                        if not plane_names:
                            logger.warning(
                                f"Keine Planes in 405-Kanal für Zelle {cell_name}"
                            )
                            failed_count += 1
                            continue

                        # Bestimme zu verarbeitende Planes
                        if plane_selection == "middle":
                            selected_planes = [plane_names[len(plane_names) // 2]]
                        elif plane_selection == "first":
                            selected_planes = [plane_names[0]]
                        elif plane_selection == "last":
                            selected_planes = [plane_names[-1]]
                        elif plane_selection == "all":
                            selected_planes = plane_names
                        else:
                            selected_planes = [
                                plane_names[len(plane_names) // 2]
                            ]  # Default

                        # Sammle Daten dieser Zelle für Batch
                        cell_data = {
                            "cell_name": cell_name,
                            "cell_idx": cell_idx,
                            "nuclei_seg_group": nuclei_seg_group,
                            "selected_planes": selected_planes,
                            "images": [],
                        }

                        for plane_name in selected_planes:
                            image_405 = channel_405[plane_name][()]
                            cell_data["images"].append(image_405)

                        cell_batch_data.append(cell_data)

                        # Verarbeite Batch wenn genug Zellen gesammelt oder am Ende
                        if (
                            len(cell_batch_data) >= cells_per_batch
                            or cell_idx == total_cells - 1
                        ):
                            # Sammle alle Bilder aus allen Zellen im Batch (flatten the list)
                            batch_images = []
                            for cell_data in cell_batch_data:
                                batch_images.extend(cell_data["images"])

                            batch_num = (processed_count // cells_per_batch) + 1

                            # Debug: Check batch_images structure
                            logger.info(
                                f"Segmentiere Batch {batch_num}: {len(cell_batch_data)} Zellen, "
                                f"{len(batch_images)} Planes gesamt"
                            )
                            logger.debug(
                                f"batch_images type: {type(batch_images)}, "
                                f"first element type: {type(batch_images[0]) if batch_images else 'empty'}, "
                                f"first element shape: {batch_images[0].shape if batch_images and hasattr(batch_images[0], 'shape') else 'N/A'}"
                            )

                            # Batch-Segmentierung aller Planes
                            # IMPORTANT: Cellpose v4 design for 2D images:
                            # - Pass as LIST: processes sequentially, but GPU-parallelizes within each image
                            # - batch_size: number of 256x256 patches processed in parallel per image
                            # - Passing (N,H,W) array: treats N as Z-dimension for 3D, NOT multiple 2D images!
                            #
                            # Performance optimization: Increase batch_size for better GPU utilization
                            start_time = time.time()

                            logger.info(
                                f"Segmentiere {len(batch_images)} Planes "
                                f"(sequential mit GPU-parallelisierung, batch_size=64 patches)"
                            )

                            masks_batch, flows_batch, styles_batch = model.eval(
                                batch_images,  # Keep as list for 2D processing
                                diameter=diameter,
                                batch_size=64,  # Higher = more 256x256 patches in parallel per image
                            )

                            elapsed = time.time() - start_time
                            logger.info(
                                f"Batch segmentiert in {elapsed:.2f}s "
                                f"({elapsed/len(batch_images):.3f}s/plane, "
                                f"{elapsed/len(cell_batch_data):.3f}s/cell)"
                            )

                            # Debug: Check masks_batch structure
                            logger.info(
                                f"masks_batch type: {type(masks_batch)}, "
                                f"shape: {masks_batch.shape if hasattr(masks_batch, 'shape') else f'len={len(masks_batch)}'}, "
                                f"dtype: {masks_batch.dtype if hasattr(masks_batch, 'dtype') else 'N/A'}"
                            )
                            if (
                                hasattr(masks_batch, "shape")
                                and len(masks_batch.shape) >= 2
                            ):
                                logger.info(
                                    f"First mask unique values: {np.unique(masks_batch[0])[:10]}"
                                )
                                logger.info(
                                    f"First mask nuclei count: {len(np.unique(masks_batch[0])) - 1}"
                                )

                            # Verteile Ergebnisse zurück zu den Zellen
                            mask_idx = 0
                            for cell_data in cell_batch_data:
                                cell_name = cell_data["cell_name"]
                                cell_idx_local = cell_data["cell_idx"]
                                nuclei_seg_group = cell_data["nuclei_seg_group"]
                                selected_planes = cell_data["selected_planes"]
                                cell_images = cell_data["images"]

                                # Speichere Segmentierungsergebnisse für alle Planes dieser Zelle
                                cell_nuclei_count = 0
                                for plane_idx, plane_name in enumerate(selected_planes):
                                    masks = masks_batch[mask_idx]
                                    image_405 = cell_images[plane_idx]
                                    mask_idx += 1

                                    # Speichere Segmentierungsmaske als uint16 mit Kompression
                                    nuclei_seg_group.create_dataset(
                                        plane_name,
                                        data=masks.astype(np.uint16),
                                        compression="gzip",
                                        compression_opts=6,
                                        shuffle=True,
                                    )

                                    # Zähle Nukleai
                                    nuclei_in_plane = (
                                        len(np.unique(masks)) - 1
                                    )  # -1 für Hintergrund
                                    cell_nuclei_count += nuclei_in_plane
                                    total_nuclei += nuclei_in_plane

                                    logger.debug(
                                        f"  {cell_name}/{plane_name}: {nuclei_in_plane} Nukleai"
                                    )

                                    # Visualisierung (optional, nur erste 3 Zellen)
                                    if visualize and cell_idx_local < 3:
                                        visualize_segmentation_result(
                                            image_405,
                                            masks,
                                            f"{cell_name}_{plane_name}",
                                        )

                                # Speichere Zell-Metadaten
                                nuclei_seg_group.attrs["nuclei_count"] = (
                                    cell_nuclei_count
                                )
                                nuclei_seg_group.attrs["planes_processed"] = len(
                                    selected_planes
                                )

                                processed_count += 1

                            # Leere Batch für nächste Iteration
                            cell_batch_data = []
                            logger.info(
                                f"Progress: {processed_count}/{total_cells} cells processed, "
                                f"{total_nuclei} nuclei found so far"
                            )

                    except Exception as e:
                        logger.error(
                            f"Fehler bei Vorbereitung von Zelle {cell_name}: {e}"
                        )
                        failed_count += 1
                        # Entferne fehlgeschlagene Zelle aus Batch
                        cell_batch_data = [
                            c for c in cell_batch_data if c["cell_name"] != cell_name
                        ]
                        continue

    except Exception as e:
        logger.error(f"Critical error during segmentation: {e}")
        logger.error("H5 files will be properly closed")
        raise
    finally:
        logger.info("Ensuring all resources are properly released")

    # Zusammenfassung
    logger.info("=" * 50)
    logger.info("SEGMENTIERUNG ABGESCHLOSSEN")
    logger.info("=" * 50)
    logger.info(f"Verarbeitete Zellen: {processed_count}")
    logger.info(f"Fehlgeschlagene Zellen: {failed_count}")
    logger.info(f"Gefundene Nukleai gesamt: {total_nuclei}")
    if processed_count > 0:
        logger.info(f"Durchschnitt: {total_nuclei/processed_count:.1f} Nukleai/Zelle")
    logger.info(f"Output gespeichert: {output_h5_path}")

    return {
        "processed_cells": processed_count,
        "failed_cells": failed_count,
        "total_nuclei": total_nuclei,
        "output_path": output_h5_path,
    }


def visualize_segmentation_result(image, masks, title):
    """
    Visualisiert ein Segmentierungsergebnis.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Original Bild
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title(f"{title}\nOriginal 405")
    axes[0].axis("off")

    # Segmentierungsmasken
    axes[1].imshow(masks, cmap="tab10")
    nuclei_count = len(np.unique(masks)) - 1
    axes[1].set_title(f"Segmentierung\n{nuclei_count} Nukleai")
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(image, cmap="gray", alpha=0.7)
    axes[2].imshow(masks, cmap="tab10", alpha=0.3)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def verify_nuclei_segmentation(h5_path, sample_cells=3):
    """
    Überprüft die nuclei_seg Daten in einer H5-Datei.

    Parameter:
    - h5_path: Pfad zur H5-Datei
    - sample_cells: Anzahl Zellen zur Stichprobenprüfung
    """
    logger.info(f"Überprüfe nuclei_seg in: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        cell_names = list(f.keys())
        cells_with_nuclei_seg = 0
        total_nuclei = 0

        for i, cell_name in enumerate(cell_names[:sample_cells]):
            cell_group = f[cell_name]
            logger.info(f"\nZelle {cell_name}:")
            logger.info(f"  Kanäle: {list(cell_group.keys())}")

            if "nuclei_seg" in cell_group:
                cells_with_nuclei_seg += 1
                nuclei_group = cell_group["nuclei_seg"]
                planes = list(nuclei_group.keys())

                logger.info(f"  nuclei_seg: {len(planes)} Planes")

                # Metadaten
                if nuclei_group.attrs:
                    logger.info("  Metadaten:")
                    for key in nuclei_group.attrs.keys():
                        logger.info(f"    {key}: {nuclei_group.attrs[key]}")

                # Prüfe erste Plane
                if planes:
                    first_plane_data = nuclei_group[planes[0]][()]
                    unique_labels = np.unique(first_plane_data)
                    nuclei_count = len(unique_labels) - 1
                    total_nuclei += nuclei_count

                    logger.info(
                        f"    Plane {planes[0]}: {first_plane_data.shape}, "
                        f"{nuclei_count} Nukleai, dtype: {first_plane_data.dtype}"
                    )
            else:
                logger.info(f"  ❌ Keine nuclei_seg gefunden")

        logger.info(f"\nZUSAMMENFASSUNG:")
        logger.info(f"Zellen mit nuclei_seg: {cells_with_nuclei_seg}/{len(cell_names)}")
        logger.info(f"Nukleai in Stichprobe: {total_nuclei}")


def main():
    parser = argparse.ArgumentParser(
        description="Nukleisegmentierung mit Cellpose - Direkte H5-Verarbeitung"
    )
    parser.add_argument(
        "--path", type=str, required=True, help="Pfad zur Input H5-Datei"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Pfad zur Output H5-Datei (default: neue Datei in /myhome/iris)",
    )
    parser.add_argument(
        "--gpu", action="store_true", help="GPU für Segmentierung verwenden"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Segmentierungsergebnisse visualisieren",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="nuclei",
        choices=["nuclei", "cyto", "cyto2"],
        help="Cellpose-Modelltyp (default: nuclei)",
    )
    parser.add_argument(
        "--diameter",
        type=int,
        default=None,
        help="Nukleaidurchmesser in Pixeln (default: automatisch)",
    )
    parser.add_argument(
        "--max_cells",
        type=int,
        default=None,
        help="Maximale Anzahl zu verarbeitender Zellen",
    )
    parser.add_argument(
        "--plane_selection",
        type=str,
        default="all",
        choices=["middle", "first", "last", "all"],
        help="Welche Planes verarbeiten (default: all)",
    )
    parser.add_argument(
        "--cells_per_batch",
        type=int,
        default=10,
        help="Anzahl Zellen pro Batch für Inferenz (default: 10, höher=schneller aber mehr GPU Memory)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Bestehende nuclei_seg überschreiben"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="nuclei_seg Daten nach Verarbeitung überprüfen",
    )

    args = parser.parse_args()

    # Setup logging
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(
        log_dir, f"segment_nuclei_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Überprüfe ob Input-Datei existiert
    if not os.path.exists(args.path):
        logger.error(f"Input-Datei nicht gefunden: {args.path}")
        return 1

    try:
        # Führe Segmentierung durch
        logger.info("Starte Nukleisegmentierung...")
        results = segment_nuclei_cellpose_direct(
            h5_path=args.path,
            output_h5_path=args.output_path,
            gpu=args.gpu,
            visualize=args.visualize,
            model_type=args.model_type,
            diameter=args.diameter,
            max_cells=args.max_cells,
            plane_selection=args.plane_selection,
            overwrite=args.overwrite,
            cells_per_batch=args.cells_per_batch,
        )

        logger.info("Segmentierung erfolgreich abgeschlossen!")

        # Überprüfung (falls gewünscht)
        if args.verify:
            logger.info("Überprüfe nuclei_seg Daten...")
            verify_nuclei_segmentation(results["output_path"])

        return 0

    except Exception as e:
        logger.error(f"Fehler bei Segmentierung: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
