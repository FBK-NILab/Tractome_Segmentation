# Tractome_Segmentation
Generation of segmentation session suitable for Tractome toolkit

The script creates a .seg file and a .spa file.
After running the script load the seg file using 'Load Segmentation' in Tractome 

## Example
```
python generate_tractome_seg.py -struct example_data/struct.nii.gz \
                                -T example_data/tract.trk \
                                -b example_data/seg.trk \
                                -o ifof_part.seg
```

By default the script computes a superset of seg.trk using a radius nearest-neaighbor, with r=10mm. To change the nn parameter use -r <radius> or -k <number of neighbors>

