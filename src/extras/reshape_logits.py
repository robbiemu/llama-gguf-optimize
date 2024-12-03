import h5py
from tqdm import tqdm  # Import tqdm for progress bars


def process_logits(source_file, output_file, chunks, context_size, n_vocab):
    with h5py.File(source_file, "r") as src:
        # Extract metadata and validate
        src_logits = src["logits"]
        src_total_chunks, src_n_ctx, src_n_vocab = src_logits.shape

        print(f"Source logits shape: {src_logits.shape}")
        #print(f"Sample source data (chunk 0): {src_logits[0, :10, :10]}")  # Debug source data

        if context_size > src_n_ctx or n_vocab > src_n_vocab:
            raise ValueError(
                f"Requested (context_size={context_size}, n_vocab={n_vocab}) exceeds source logits dimensions "
                f"(context_size={src_n_ctx}, n_vocab={src_n_vocab})."
            )
        if src_n_ctx % context_size != 0 or src_n_vocab % n_vocab != 0:
            raise ValueError(
                f"(context_size={context_size}, n_vocab={n_vocab}) must divide source logits dimensions "
                f"(context_size={src_n_ctx}, n_vocab={src_n_vocab})."
            )

        # Calculate the number of slices along n_vocab and context_size
        n_vocab_slices = src_n_vocab // n_vocab
        n_ctx_slices = src_n_ctx // context_size

        # Calculate total possible output chunks
        total_output_chunks = src_total_chunks * n_ctx_slices * n_vocab_slices

        if chunks != total_output_chunks:
            print(
                f"Warning: Number of requested chunks ({chunks}) does not match the total available slices ({total_output_chunks}).\n"
                f"Proceeding to process up to the minimum of both."
            )
            chunks = min(chunks, total_output_chunks)

        # Prepare output
        with h5py.File(output_file, "w") as dest:
            dest.attrs["format"] = "processed_logits"
            dest.attrs["n_ctx"] = context_size
            dest.attrs["n_vocab"] = n_vocab
            dest.attrs["total_chunks"] = chunks

            logits_dset = dest.create_dataset(
                "logits", (chunks, context_size, n_vocab), dtype="f4"
            )
            chunk_index_dset = dest.create_dataset("chunk_index", (chunks,), dtype="i8")
            processed_chunks_dset = dest.create_dataset(
                "processed_chunks", (chunks,), dtype="i1", fillvalue=0
            )

            output_chunk_idx = 0

            # Initialize the progress bar
            with tqdm(total=chunks, desc="Processing Chunks", unit="chunk") as pbar:
                for src_chunk_idx in range(src_total_chunks):
                    for ctx_slice_idx in range(n_ctx_slices):
                        for vocab_slice_idx in range(n_vocab_slices):
                            if output_chunk_idx >= chunks:
                                break

                            # Define the slicing indices
                            ctx_start = ctx_slice_idx * context_size
                            ctx_end = ctx_start + context_size
                            vocab_start = vocab_slice_idx * n_vocab
                            vocab_end = vocab_start + n_vocab

                            # Extract the slice from the source logits
                            input_chunk_data = src_logits[src_chunk_idx, ctx_start:ctx_end, vocab_start:vocab_end]

                            # Reshape and write to output
                            reshaped_chunk = input_chunk_data  # Already the desired shape

                            logits_dset[output_chunk_idx] = reshaped_chunk
                            chunk_index_dset[output_chunk_idx] = output_chunk_idx
                            processed_chunks_dset[output_chunk_idx] = 0

                            output_chunk_idx += 1
                            pbar.update(1)  # Update the progress bar

            print(f"Total processed chunks: {output_chunk_idx}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reshape logits into specified chunks.")
    parser.add_argument("destination", type=str, nargs='?', default="output_logits.h5",
                        help="output file (defaults to output_logits.h5)")
    parser.add_argument("--source", required=True, help="Path to the source logits file.")
    parser.add_argument("--chunks", type=int, required=True, help="Number of chunks in the output.")
    parser.add_argument("--context-size", type=int, required=True, help="Context size of each chunk.")
    parser.add_argument("--n-vocab", type=int, required=True, help="Vocab size of each chunk.")

    args = parser.parse_args()

    process_logits(
        source_file=args.source,
        output_file=args.destination,
        chunks=args.chunks,
        context_size=args.context_size,
        n_vocab=args.n_vocab,
    )
