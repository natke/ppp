from pathlib import Path
import onnx
from transformers import MobileBertForSequenceClassification
import pre_post_processing_poc_WIP as ppp


class TokenizeForBert(ppp.Step):
    def __init__(self, name: str = None):
        super().__init__(['text'], ['input_ids', 'token_type_ids', 'attention_mask'], name)

    def _create_graph_for_step(self, graph: onnx.GraphProto):
        input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, 0)
        output_shape_str = f'tokenize_ppp_{self.step_num}_h, tokenize_ppp_{self.step_num}_w, tokenize_ppp_{self.step_num}_c'

        converter_graph = onnx.parser.parse_graph(f'''\
            tokenize ({input_type_str}[{input_shape_str}] {self.input_names[0]}) 
                => ({input_type_str}[{output_shape_str}C] {self.output_names[0]})  
            {{
                {self.output_names[0]} = ortext.Tokenize ({self.input_names[0]})
            }}
            ''')

        onnx.checker.check_graph(converter_graph, ppp.Step._custom_op_checker_context)
        return converter_graph

def bert(model_file: Path, output_file: Path):
    model = onnx.load(str(model_file.resolve(strict=True)))

    # TODO Implement this
    inputs = [ppp.create_value_info_for_text('text')]

    model_input_shape = model.graph.input[0].type.tensor_type.shape
    model_output_shape = model.graph.output[0].type.tensor_type.shape
    assert(model_input_shape.dim[-1].HasField("dim_value"))
    assert(model_input_shape.dim[-2].HasField("dim_value"))
    assert(model_output_shape.dim[-1].HasField("dim_value"))
    assert(model_output_shape.dim[-2].HasField("dim_value"))

    pipeline = ppp.PrePostProcessor(inputs)
    pipeline.add_pre_processing([
        TokenizeForBert(),  
        ppp.Unsqueeze([0]),    # add batch to match original model
    ])

    # TODO Add post processing

    new_model = pipeline.run(model)
    onnx.save_model(new_model, str(output_file.resolve()))


mobilebert_path = r'lordtt13-emo-mobilebert.onnx'
mobilebert_aug_path = r'lordtt13-emo-mobilebert-aug.onnx'

def update_mobilebert():
    ppp.bert(Path(mobilebert_path), Path(mobilebert_aug_path))
