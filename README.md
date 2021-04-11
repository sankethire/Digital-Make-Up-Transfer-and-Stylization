# Digital-Make-Up-Transfer-and-Stylization

Implementation of [Digital face makeup by example](https://ieeexplore.ieee.org/document/5206833) with the following modifications -

1. Face key point detector
2. Bilateral filtering is used instead of weighted-least-square (WLS)

## How to run?

Place `shape_predictor_68_face_landmarks.dat` in src directory in not already there. You can download it from [dlib official site](http://dlib.net/files/)

## Design

The entire process can be broken into following steps -

### Face Alignment

### Layer Decomposition

### Skin Detail Transfer

### Color Transfer

### Highlight and Shading Transfer

### Lip Makeup


## File Structure

1. `digital_makeup.py`
   1. Main py file, provides command line interface to call the entire digital makeup transfer process
2. `src/face_alignment.py`
3. `src/skin_detector.py`

## References

1. [Similar implementation](https://github.com/TheMathWizard/Face-Makeup-by-Example)
2. Put other bookmarked links
