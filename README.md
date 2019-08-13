# Tractome_Segmentation
Generation of segmentation session suitable for Tractome toolkit

## Example
`python generate_tractome_seg.py -struct example_data/struct.nii.gz \
                                -T example_data/tract.trk \
                                -b example_data/seg.trk \
                                -o ifof_part.seg`

By default the script compute a superset of seg.trk using a radius nearest-neaighbor, with r=10mm. To change the nn parameter use -r <radius> or -k <number of neighbors>
