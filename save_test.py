import unittest
import tempfile
import torch 

from src.models import CoCoCoOp
from src.datasets import Caltech101

class TestCoCoCoOpSaveLoad(unittest.TestCase):

    def test(self):
        ds = Caltech101(split='val', cache_transformed_images=True, download=True)
        ds.one_hot_encode_labels()
        

        I = 3

        kwargs = {
            'classnames' : ds.get_class_names(),
            'clip_model_name' : 'ViT-B/16',
            'ctx_init' : "a photo of a",
            'n_ctx' : 10,
            'prec' : "fp32"
        }
        model =  CoCoCoOp()
        model.build_model(**kwargs)
        model.model.eval()

        ds.transform = model.img_to_features

        results = []
        with torch.no_grad():
            for i in range(I):

                result = model.forward(ds[i][0].unsqueeze(0))
                results.append(result)
        
        with tempfile.NamedTemporaryFile() as f:
            
            model.save_model(f.name)

            model = CoCoCoOp()
            model.build_model(**kwargs)
            model.load_model(f.name)
            model.model.eval()

            ds.transform = model.img_to_features

            with torch.no_grad():
                for i in range(I):
                    result = model.forward(ds[i][0].unsqueeze(0))
                    self.assertTrue(torch.allclose(results[i], result))


if __name__ == '__main__':
    unittest.main()