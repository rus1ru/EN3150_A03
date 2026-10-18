"""Architecture budget, learning signal and deterministic recovery for Model B v2."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from en3150_a03.models.model_b_v2 import ModelBV2
from en3150_a03.model_b_measure import profile_model
from en3150_a03.model_b_v2_train import train, seed_all, learning_rate


class ModelBV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_budget_depthwise_shapes_and_gradients(self):
        for attention, expected in [(True,96538),(False,87546)]:
            model=ModelBV2(attention=attention)
            self.assertEqual(sum(p.numel() for p in model.parameters()),expected)
            for module in model.modules():
                if isinstance(module,torch.nn.Conv2d) and module.kernel_size==(3,3):
                    self.assertEqual(module.groups,module.in_channels)
            logits=model(torch.randn(2,3,64,64))
            self.assertEqual(tuple(logits.shape),(2,10))
            torch.nn.functional.cross_entropy(logits,torch.tensor([0,9])).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertEqual(profile_model(ModelBV2())["macs"],21358976)

    def loaders(self):
        x=torch.arange(8*3*16*16,dtype=torch.float32).reshape(8,3,16,16)/(8*3*16*16)
        ds=TensorDataset(x,torch.tensor([0,0,0,0,1,1,1,1]))
        return {name:DataLoader(ds,batch_size=4,shuffle=name=="train",generator=torch.Generator().manual_seed(42)) for name in ["train","val"]}

    def test_dropout_batchnorm_and_optimizer_resume_exactly(self):
        cfg=dict(optimizer="adamw",learning_rate=.003,weight_decay=1e-4,amp=False,epochs=3,seed=42,label_smoothing=.05)
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            root=Path(d); (root/"full").mkdir(); (root/"resume").mkdir()
            seed_all(42)
            train(ModelBV2(2),self.loaders(),torch.device("cpu"),cfg,root/"full")
            seed_all(42)
            train(ModelBV2(2),self.loaders(),torch.device("cpu"),cfg,root/"resume",stop_after=1)
            train(ModelBV2(2),self.loaders(),torch.device("cpu"),cfg,root/"resume",resume=True)
            a=torch.load(root/"full/last_model.pt",weights_only=True)
            b=torch.load(root/"resume/last_model.pt",weights_only=True)
            self.assertEqual([r['val_loss'] for r in a['history']],[r['val_loss'] for r in b['history']])
            for key in a['model_state_dict']:
                self.assertTrue(torch.equal(a['model_state_dict'][key],b['model_state_dict'][key]),key)
            self.assertFalse((root/"full/test_metrics.json").exists())

    def test_cosine_schedule(self):
        self.assertAlmostEqual(learning_rate(1,60,.003),.0006)
        self.assertAlmostEqual(learning_rate(5,60,.003),.003)
        self.assertAlmostEqual(learning_rate(60,60,.003),1e-5)
        with self.assertRaises(ValueError): ModelBV2(100)

if __name__=="__main__": unittest.main()
