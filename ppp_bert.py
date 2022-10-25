from pathlib import Path
import onnx
import pre_post_processing_poc as ppp



class TokenizeForBert(ppp.Step):
    def __init__(self, name: str = None):
        super().__init__(['text'], ['input_ids', 'token_type_ids', 'attention_mask'], name)

    def _create_graph_for_step(self, graph: onnx.GraphProto):
        input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, 0)
        assert(input_type_str == 'string')
        output_shape_str = f'1, tokenize_ppp_{self.step_num}_numids'

        converter_graph = onnx.parser.parse_graph(f'''\
            tokenize ({input_type_str}[{input_shape_str}] {self.input_names[0]}) 
                => (int64[{output_shape_str}] {self.output_names[0]}, 
                    int64[{output_shape_str}] {self.output_names[1]}, 
                    int64[{output_shape_str}] {self.output_names[2]})  
            {{
                {self.output_names[0]}, {self.output_names[1]}, {self.output_names[2]} = 
                    ai.onnx.contrib.BertTokenizer <strip_accents=0, do_lower_case=1> ({self.input_names[0]})
            }}
            ''')

        vocab_file = converter_graph.node[0].attribute.add()
        vocab_file.name = "vocab_file"
        vocab_file.type = onnx.AttributeProto.AttributeType.STRING
        vocab_file.s = open('bert_basic_cased_vocab.txt', 'rb').read()

        onnx.checker.check_graph(converter_graph, self._custom_op_checker_context)

        return converter_graph


def bert(model_file: Path, output_file: Path):
    model = onnx.load(str(model_file.resolve(strict=True)))

    # create a ValueInfoProto for a buffer of bytes containing an input image. could be jpeg/png/bmp
    input_type = onnx.helper.make_tensor_type_proto(elem_type=onnx.TensorProto.STRING, shape=['max_seq_length'])
    inputs = [onnx.helper.make_value_info('text', input_type)]

    pipeline = ppp.PrePostProcessor(inputs)
    pipeline.add_pre_processing([
        TokenizeForBert(),  # this outputs 3 values, each with a batch dim of 1 which
    ])

    # TODO Add post processing

    new_model = pipeline.run(model)
    onnx.save_model(new_model, str(output_file.resolve()))
