# evaluate/shap_explainer.py
import shap

def explain_with_shap(model, dataloader, device='cpu'):
    model.eval()
    X, y = [], []
    for batch in dataloader:
        X.append(batch['features'])
        y.append(batch['label'])
    X = torch.cat(X).to(device)
    y = torch.cat(y).to(device)

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    shap.summary_plot(shap_values, X)
