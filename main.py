import pre_post_processing_poc_WIP as ppp
import ppp_bert
from pathlib import Path

mobilenet_path = r'mobilenetv2-7.onnx'
mobilenet_withppp_path = r'mobilenetv2-7-aug.onnx'

def my_update_mobilenet():
    ppp.mobilenet(Path(mobilenet_path), Path(mobilenet_withppp_path))

def main():
    my_update_mobilenet()
    ppp_bert.update_mobilebert()
 
if __name__ == '__main__':
    main()
