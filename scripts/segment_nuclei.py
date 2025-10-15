import argparse
import h5py
import numpy as np
from cellpose import models
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
    model_type="nuclei",
    diameter=None,
    max_cells=None,
    plane_selection="middle",
    overwrite=False,
    cells_per_batch=2,
    log_file=None,
):
    """
    Segments nuclei directly from H5 file and adds nuclei_seg channel.
    Processes cells in batches (default: 2 cells), batching all planes together.

    Parameters:
    - h5_path: Path to input H5 file
    - output_h5_path: Path to output H5 file (None = edits input file directly)
    - gpu: Use GPU
    - visualize: Visualize segmentation results
    - model_type: Cellpose model type ('nuclei', 'cyto', 'cyto2')
    - diameter: Nucleus diameter (None = automatic)
    - max_cells: Maximum number of cells to process
    - plane_selection: Which planes to process ('middle', 'first', 'last', 'all')
    - overwrite: Overwrite existing nuclei_seg
    - cells_per_batch: Number of cells per batch (default: 2)
    - log_file: Path to log file for tqdm output (None = stderr)
    """

    global _output_file_path

    _output_file_path = output_h5_path
    mode = "w"  # Always create new file

    # Setup tqdm file output
    tqdm_file = None
    if log_file:
        try:
            tqdm_file = open(log_file, "a")
            logger.info(f"tqdm progress will be logged to: {log_file}")
        except Exception as e:
            logger.warning(f"Could not open tqdm log file: {e}")
            tqdm_file = sys.stderr
    else:
        tqdm_file = sys.stderr

    # Check available disk space
    import shutil

    output_dir = os.path.dirname(output_h5_path)
    free_bytes = shutil.disk_usage(output_dir).free
    free_gb = free_bytes / (1024**3)

    if free_gb < 1.0:
        raise RuntimeError(
            f"Not enough disk space in {output_dir}. Available: {free_gb:.2f} GB"
        )

    logger.info(f"Available disk space in {output_dir}: {free_gb:.2f} GB")

    # Load Cellpose model with timeout handling
    logger.info(f"Loading Cellpose model: {model_type}, GPU: {gpu}")
    try:
        model = models.CellposeModel(gpu=gpu, model_type=model_type)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading Cellpose model: {e}")
        logger.error("Tip: For network issues, download model manually beforehand")
        raise

    # Metadata for segmentation
    segmentation_metadata = {
        "cellpose_model": model_type,
        "diameter": diameter if diameter else "auto",
        "segmentation_date": datetime.now().isoformat(),
        "plane_selection": plane_selection,
        "source_file": os.path.basename(h5_path),
    }

    # Open new H5 file and copy + extend data
    # Using context managers ensures files are properly closed even on exceptions
    try:
        with h5py.File(output_h5_path, mode) as output_file:

            # Copy all original data from input file
            logger.info("Copying original data structure...")
            with h5py.File(h5_path, "r") as input_file:

                # Add metadata to output file
                output_file.attrs["source_file"] = os.path.basename(h5_path)
                output_file.attrs["creation_date"] = datetime.now().isoformat()
                output_file.attrs["purpose"] = "complete_with_nuclei_segmentation"

                cell_names = list(input_file.keys())
                total_cells = (
                    min(len(cell_names), max_cells) if max_cells else len(cell_names)
                )

                logger.info(
                    f"Processing {total_cells} cells from {len(cell_names)} available"
                )

                # Process each cell - first collect all data for batch processing
                processed_count = 0
                failed_count = 0

                # Process cells in batches (e.g. 2 cells simultaneously)
                cell_batch_data = []  # Collects data for multiple cells

                for cell_idx, cell_name in enumerate(
                    tqdm(
                        cell_names[:total_cells],
                        desc="Copy & Segment Cells",
                        file=tqdm_file,
                        mininterval=1.0,
                    )
                ):
                    try:
                        # Copy original cell data
                        input_file.copy(cell_name, output_file)
                        cell_group = output_file[cell_name]

                        if "405" not in cell_group:
                            logger.warning(f"No 405 channel in cell {cell_name}")
                            failed_count += 1
                            continue

                        # Check if nuclei_seg already exists
                        if "nuclei_seg" in cell_group and not overwrite:
                            logger.info(
                                f"nuclei_seg already present for {cell_name}, skipping"
                            )
                            processed_count += 1
                            continue

                        # Create/overwrite nuclei_seg group
                        if "nuclei_seg" in cell_group:
                            del cell_group["nuclei_seg"]
                        nuclei_seg_group = cell_group.create_group("nuclei_seg")

                        # Add metadata
                        for key, value in segmentation_metadata.items():
                            nuclei_seg_group.attrs[key] = value

                        # Load 405 channel data
                        channel_405 = cell_group["405"]
                        plane_names = list(channel_405.keys())

                        if not plane_names:
                            logger.warning(
                                f"No planes in 405 channel for cell {cell_name}"
                            )
                            failed_count += 1
                            continue

                        # Determine planes to process
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

                        # Collect data for this cell for batch
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

                        # Process batch when enough cells collected or at end
                        if (
                            len(cell_batch_data) >= cells_per_batch
                            or cell_idx == total_cells - 1
                        ):
                            # Collect all images from all cells in batch (flatten the list)
                            batch_images = []
                            for cell_data in cell_batch_data:
                                batch_images.extend(cell_data["images"])

                            batch_num = (processed_count // cells_per_batch) + 1

                            # Debug: Check batch_images structure
                            logger.info(
                                f"Segmenting batch {batch_num}: {len(cell_batch_data)} cells, "
                                f"{len(batch_images)} planes total"
                            )
                            logger.debug(
                                f"batch_images type: {type(batch_images)}, "
                                f"first element type: {type(batch_images[0]) if batch_images else 'empty'}, "
                                f"first element shape: {batch_images[0].shape if batch_images and hasattr(batch_images[0], 'shape') else 'N/A'}"
                            )

                            start_time = time.time()

                            logger.info(
                                f"Segmenting {len(batch_images)} planes "
                                f"(sequential with GPU parallelization, batch_size=64 patches)"
                            )

                            masks_batch, flows_batch, styles_batch = model.eval(
                                batch_images,
                                diameter=diameter,
                                batch_size=64,
                            )

                            elapsed = time.time() - start_time
                            logger.info(
                                f"Batch segmented in {elapsed:.2f}s "
                                f"({elapsed/len(batch_images):.3f}s/plane, "
                                f"{elapsed/len(cell_batch_data):.3f}s/cell)"
                            )

                            # Debug: Check masks_batch structure
                            logger.info(
                                f"masks_batch type: {type(masks_batch)}, "
                                f"shape: {masks_batch.shape if hasattr(masks_batch, 'shape') else f'len={len(masks_batch)}'}, "
                                f"dtype: {masks_batch.dtype if hasattr(masks_batch, 'dtype') else 'N/A'}"
                            )

                            # Distribute results back to cells
                            mask_idx = 0
                            for cell_data in cell_batch_data:
                                cell_name = cell_data["cell_name"]
                                cell_idx_local = cell_data["cell_idx"]
                                nuclei_seg_group = cell_data["nuclei_seg_group"]
                                selected_planes = cell_data["selected_planes"]
                                cell_images = cell_data["images"]

                                # Save segmentation results for all planes of this cell
                                for plane_idx, plane_name in enumerate(selected_planes):
                                    masks = masks_batch[mask_idx]
                                    image_405 = cell_images[plane_idx]
                                    mask_idx += 1

                                    # Save segmentation mask as uint16 with compression
                                    nuclei_seg_group.create_dataset(
                                        plane_name,
                                        data=masks.astype(np.uint16),
                                        compression="gzip",
                                        compression_opts=6,
                                        shuffle=True,
                                    )

                                # Save cell metadata
                                nuclei_seg_group.attrs["planes_processed"] = len(
                                    selected_planes
                                )

                                processed_count += 1

                            # Clear batch for next iteration
                            cell_batch_data = []
                            logger.info(
                                f"Progress: {processed_count}/{total_cells} cells processed"
                            )

                    except Exception as e:
                        logger.error(f"Error preparing cell {cell_name}: {e}")
                        failed_count += 1
                        # Remove failed cell from batch
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

    # Close tqdm file if it was opened
    if tqdm_file and tqdm_file != sys.stderr:
        try:
            tqdm_file.close()
        except:
            pass

    # Summary
    logger.info("=" * 50)
    logger.info("SEGMENTATION COMPLETED")
    logger.info("=" * 50)
    logger.info(f"Processed cells: {processed_count}")
    logger.info(f"Failed cells: {failed_count}")
    logger.info(f"Output saved: {output_h5_path}")

    return {
        "processed_cells": processed_count,
        "failed_cells": failed_count,
        "output_path": output_h5_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Nuclei segmentation with Cellpose - Direct H5 processing"
    )
    parser.add_argument("--path", type=str, required=True, help="Path to input H5 file")
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to output H5 file (default: new file in /myhome/iris)",
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU for segmentation")

    parser.add_argument(
        "--model_type",
        type=str,
        default="nuclei",
        choices=["nuclei", "cyto", "cyto2"],
        help="Cellpose model type (default: nuclei)",
    )
    parser.add_argument(
        "--diameter",
        type=int,
        default=None,
        help="Nucleus diameter in pixels (default: automatic)",
    )
    parser.add_argument(
        "--max_cells",
        type=int,
        default=None,
        help="Maximum number of cells to process",
    )
    parser.add_argument(
        "--plane_selection",
        type=str,
        default="all",
        choices=["middle", "first", "last", "all"],
        help="Which planes to process (default: all)",
    )
    parser.add_argument(
        "--cells_per_batch",
        type=int,
        default=10,
        help="Number of cells per batch for inference (default: 10, higher=faster but more GPU memory)",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Path to log file for tqdm progress (default: auto-generated)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing nuclei_seg"
    )

    args = parser.parse_args()

    # Setup logging
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_filename = (
        args.log_file
        if args.log_file
        else os.path.join(
            log_dir, f"segment_nuclei_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
    )
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Also log to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(console_handler)

    logger.info(f"Log file: {log_filename}")

    # Check if input file exists
    if not os.path.exists(args.path):
        logger.error(f"Input file not found: {args.path}")
        return 1

    try:
        # Run segmentation
        logger.info("Starting nuclei segmentation...")
        results = segment_nuclei_cellpose_direct(
            h5_path=args.path,
            output_h5_path=args.output_path,
            gpu=args.gpu,
            model_type=args.model_type,
            diameter=args.diameter,
            max_cells=args.max_cells,
            plane_selection=args.plane_selection,
            overwrite=args.overwrite,
            cells_per_batch=args.cells_per_batch,
            log_file=log_filename,  # Use same log file for tqdm
        )

        logger.info("Segmentation completed successfully!")

        return 0

    except Exception as e:
        logger.error(f"Error during segmentation: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
