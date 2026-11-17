# Annotation Exceptions

## Confirmed empty frames

- `jp/linear_algebra/video1/003.jpg.json`
- `jp/linear_algebra/video1/006.jpg.json`

The annotator reported that the board image and audio were too unclear to
identify and ground referential expressions reliably. These frames are
intentionally retained with no bounding boxes or mention--box relations rather
than forcing uncertain annotations. The validator's `empty_boxes` and
`empty_relations` warnings for these two files are therefore expected.
