import h5py

for name in ["baseline", "target"]:
    # Open the HDF5 file in read/write mode
    with h5py.File(f'{name}_logits.h5', 'r+') as f:
        # Delete the existing dataset
        del f['freed_chunks']
    
        # Create a new dataset with the desired properties
        f.create_dataset('freed_chunks', data=[], maxshape=(None,))

    # Verify the changes
    with h5py.File(f'{name}_logits.h5', 'r') as f:
        print(name, f['freed_chunks'][:])


