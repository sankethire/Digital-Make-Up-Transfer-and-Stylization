# Digital-Make-Up-Transfer-and-Stylization

Implementation of [Digital face makeup by example](https://ieeexplore.ieee.org/document/5206833) with the following modifications -

1. Face key point detector is `dlib`'s implementation with 69 points. Forehead points are manually adjustable and automatically added first using 3rd party skin detector implementation
2. Bilateral filtering is used instead of weighted-least-square (WLS)
3. XDoG of subject along with makeup is displayed. weighted sum of LAB values is taken.

## How to run?

1. Install required dependencies using `pip install -r requirements.txt`

2. Place `shape_predictor_68_face_landmarks.dat` in src directory in not already there. You can download it from [dlib official site](http://dlib.net/files/)

3. The program takes two inputs (i.e. Subject Image nd Example Image) and can be run as - `python digital makeup.py -lc -lf input/subject1.png input/example1.png`

4. The program has option for 3 flags:
   -  `-si` or show interactive to display output of intermediary steps like face alignment and various face masks extraction
   -  The other two flags `-lc` and `-lf` which imply:
      - `-lf` or light foundation. Skin details aren’t hidden when foundation is light. Weight of subjects’ for transferring skin detail (δI ) is 1 instead of 0
      - `-lc` or light makeup color. Weight while doing color transfer are more in favor of subject image, γ = 0.5 instead of γ = 0.8 or weight of subject color 0.5 instead of 0.2



## Design

The entire process can be broken into following steps -

- Face Alignment
- Layer Decomposition
- Skin Detail Transfer
- Color Transfer
- Highlight and Shading Transfer
- Lip Makeup

**(To understand the complete approach of the Design go through - [Report](https://github.com/sankethire/Digital-Make-Up-Transfer-and-Stylization/blob/main/Report.pdf))**


## File Structure

1. `digital_makeup.py`
   1. Main py file, provides command line interface to call the entire digital makeup transfer process
2. `src/face_alignment.py`
   1. Face landmark points extraction
   2. Face Warping
   3. Face mask extraction for various components - skin, lips, eyes, mouth
3. `src/makeup_util.py`
   1. Layer Decomposition
   2. Color and Detail Transfer
   3. Highlight and Shading Transfer
   4. Lip makeup special function
4. `src/xdog.py`
   1. Cartoonize an image, black and white output
5. `src/skin_detector.py`
   1. Skin detection for face used in automatic detection of forehead points
6. `input`
   1. Contains image files to test code with
7. `output`
   1. Output of some experiment on input images

## References

1. [Face Warping Tutorial](https://pysource.com/2019/05/09/select-and-warp-triangles-face-swapping-opencv-with-python-part-4/)
2. [Bilateral filtering for layer decomposition](https://www.geeksforgeeks.org/python-bilateral-filtering/)
3. [Face landmark point extraction](https://www.pyimagesearch.com/2019/12/16/training-a-custom-dlib-shape-predictor/)
4. [XDoG implementation help](https://github.com/CemalUnal/XDoG-Filter)
5. [Similar Digital Makeup implementation](https://github.com/TheMathWizard/Face-Makeup-by-Example)
