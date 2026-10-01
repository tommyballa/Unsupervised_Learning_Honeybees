import sys, torch
sys.path.insert(0, "experiments/single_bee")
import main_fastflow as mf
from torch.utils.data import DataLoader
from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.utilities.configurations import Split
from moviad.models.fastflow.fastflow import FastFlowTrainArgs
from moviad.utilities.evaluation.report import run_inference, compute_metrics, print_metrics
dev = torch.device("cuda:0")
model = mf.build_model(dev)
mf.load_model_state(model, torch.load(sys.argv[1], map_location="cpu"))
ds = VarroaDataset(mf.dataset_args(), "varroa", Split.TEST)
dl = DataLoader(ds, batch_size=16, num_workers=12, pin_memory=True)
maps, scores, labels, masks = run_inference(model, dl, dev)
print_metrics(compute_metrics(maps, scores, labels, masks, with_pro=False))
# smoke test training
tr = VarroaDataset(mf.dataset_args(), "varroa", Split.TRAIN)
opt = torch.optim.Adam(model.parameters(), lr=1e-5)
ta = FastFlowTrainArgs(epochs=1, batch_size=8, optimizer=opt); ta.init_train(model)
model.train()
x = next(iter(DataLoader(tr, batch_size=8)))
print("train_step loss", model.train_step(x, ta))
