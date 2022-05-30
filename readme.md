## 	Furigana Detection

Furigana is a part of Japanese written language. 
Japanese uses both a phonetic (representing sounds, called Hiragana) alphabet and a logographic (representing meaning, called Kanji) alphabet.
In written Japanese, the two are mixed to form sentences.
For Kanji, since the characters represent meaning, the reader may not always know how it is pronounced. 
Therefore, writers may sometimes add notes next to kanji to indicate their pronunciation, these types of notes are called Furigana. 
Furigana is typically written in the Hiragana alphabet.

Furigana can be problematic for systems that process text within images.
Furigana does not change the meaning of the text and can thus be disregarded by computers for most purposes. 
For example, current Optical Character Reading systems do not handle furigana well. 
The furigana is mistaken as regular text and inserted into the output, which significantly reduces the quality of the result. 

This project aims to detect the location of furigana in images for further processing.

## installation
- install CUDA according to tensorflow requirements: https://www.tensorflow.org/install/gpu

- install tesserocr according to installation guidelines: https://github.com/sirfz/tesserocr/tree/310ae9a09ca1105652741539e454219da7c936a1#installation  

- download tessdata (optional): https://github.com/tesseract-ocr/tessdata 

- create a folder called "data" in lib/ComicTextDetector and add comictextdetector.pt and comictextdetector.pt.onnx to it from: https://github.com/dmMaze/comic-text-detector#readme

- install pytorch with cuda support: https://pytorch.org/get-started/locally/

- install opencv-python, e.g. by running: pip install opencv-python

- install packages from requirements.txt 

## running
The program can be run from the commandline, the following command detects furigana in an image and creates a file with
the predictions. The --debug command shows the predictions in a window.

    furigana_detection.py --image "../data/example.jpg" --debug

Output is given in COCO object detection format

See the --help command for more arguments

        furigana_detection.py --help
        optional arguments:
            -h, --help            show this help message and exit
            --debug               Show debug images (result)
            --debug_area          Show debug images for each individual text area
            --folder FOLDER       Detect all images in a folder
            --image IMAGE         An image to detect furigana in
            --config CONFIG       Path to a config file with advanced configurations
            --out OUT             name of output file
            --labels LABELS       Path to ground truth labels
            --predictions PREDICTIONS
                                Path to predictions for evaluation (labels must also
                                be specified)
            --validate            validate detections using ocr (tessdata must be
                                specified)
            --eval                run evaluation. If a folder is detected, these results
                                will be evaluated



Alternatively, use the FuriganaDetector class to make detections
        
        import detection
        FuriganaDetector(verbose=True).detect(r"../data/example.jpg") 
    
## Notice - images
Akkera Kanjinchou © Yuki Kobayashi. Original images were taken from the manga109 dataset http://www.manga109.org  
example.jpg is from Tsundere Akuyaku Reijō Rīzerotte to Jikkyō no Endō-kun to Kaisetsu no Kobayashi-san © Suzu Enoshima
