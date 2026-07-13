import nbformat as nbf
import sys

nb_file = 'baseline_anomaly_detection.ipynb'
try:
    with open(nb_file, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)
except Exception as e:
    print(e)
    sys.exit(1)

# Markdown cell
md_content = """## 6. SOTA Approach: Self-Supervised Machine-ID Classification (Outlier Exposure)

As discussed in ML research methodologies, relying solely on an Autoencoder's reconstruction error might not be the most sensitive way to detect anomalies. A State-of-the-Art (SOTA) approach for DCASE 2020 Task 2 reformulates the unsupervised problem into a supervised classification problem.

Since the training set consists of normal sounds from multiple distinct machines (e.g., Machine IDs 00, 02, 04), we can train a classifier to predict the **Machine ID**. 
- **Normal sounds** will produce highly confident predictions for their respective machine ID.
- **Anomalous sounds** (which the model has never seen) will confuse the classifier, resulting in lower confidence for the correct machine ID.

The anomaly score is defined as: $1 - P(\\text{True Machine ID})$"""

nb.cells.append(nbf.v4.new_markdown_cell(md_content))

code_content_1 = """# Encode Machine IDs to integer labels
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F

le = LabelEncoder()
y_train_id = le.fit_transform(ids_train)
y_test_id = le.transform(ids_test)

# Repeat the machine ID label for each frame window in a file
# X_train_full is a list of arrays (one per file). 
y_train_frames = []
for i, x in enumerate(X_train_full):
    y_train_frames.extend([y_train_id[i]] * x.shape[0])
y_train_frames = np.array(y_train_frames)

# Split for classifier (using same train/val idx as before for consistency)
# But we need frame-level labels corresponding to X_train and X_val
y_train_frames_flat = np.hstack([[y_train_id[i]] * X_train_full[i].shape[0] for i in train_idx])
y_val_frames_flat = np.hstack([[y_train_id[i]] * X_train_full[i].shape[0] for i in val_idx])

num_classes = len(le.classes_)
print(f"Number of Machine IDs (classes): {num_classes}")
print(f"Machine IDs: {le.classes_}")
"""

nb.cells.append(nbf.v4.new_code_cell(code_content_1))

code_content_2 = """class MachineIDClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MachineIDClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

clf_model = MachineIDClassifier(FEATURE_DIM, num_classes).to(device)
clf_criterion = nn.CrossEntropyLoss()
clf_optimizer = torch.optim.Adam(clf_model.parameters(), lr=1e-3, weight_decay=1e-4)

# Create Dataloaders for Classification
train_tensor_x = torch.tensor(X_train, dtype=torch.float32)
train_tensor_y = torch.tensor(y_train_frames_flat, dtype=torch.long)
clf_train_loader = DataLoader(TensorDataset(train_tensor_x, train_tensor_y), batch_size=BATCH_SIZE, shuffle=True)

val_tensor_x = torch.tensor(X_val, dtype=torch.float32)
val_tensor_y = torch.tensor(y_val_frames_flat, dtype=torch.long)
clf_val_loader = DataLoader(TensorDataset(val_tensor_x, val_tensor_y), batch_size=BATCH_SIZE, shuffle=False)
"""
nb.cells.append(nbf.v4.new_code_cell(code_content_2))

code_content_3 = """print("Training SOTA Machine-ID Classifier...")
CLF_EPOCHS = 15
for epoch in range(CLF_EPOCHS):
    clf_model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    for batch_x, batch_y in clf_train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        clf_optimizer.zero_grad()
        outputs = clf_model(batch_x)
        loss = clf_criterion(outputs, batch_y)
        loss.backward()
        clf_optimizer.step()
        epoch_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
        
    clf_model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_x, batch_y in clf_val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = clf_model(batch_x)
            loss = clf_criterion(outputs, batch_y)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            val_total += batch_y.size(0)
            val_correct += (predicted == batch_y).sum().item()
            
    avg_train_loss = epoch_loss / len(clf_train_loader)
    train_acc = 100 * correct / total
    avg_val_loss = val_loss / len(clf_val_loader)
    val_acc = 100 * val_correct / val_total
    
    print(f"Epoch {epoch+1:02d}/{CLF_EPOCHS} | Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.2f}%")
"""
nb.cells.append(nbf.v4.new_code_cell(code_content_3))

code_content_4 = """# Evaluate the SOTA Classifier on Test Set
clf_model.eval()
sota_anomaly_scores = []
sota_true_labels = []
sota_machine_ids = []

with torch.no_grad():
    for i in tqdm(range(len(X_test_files)), desc="Scoring Test Set (SOTA)"):
        # For each file, we evaluate all its frames
        # X_test_files is a list of feature arrays (windows)!
        
        windows = X_test_files[i]
        
        input_tensor = torch.tensor(windows, dtype=torch.float32).to(device)
        outputs = clf_model(input_tensor)
        
        # Convert logits to probabilities
        probs = F.softmax(outputs, dim=1)
        
        # We know the true machine ID of this test file
        true_id = y_test_id[i]
        
        # Probability assigned to the true machine ID
        prob_true_id = probs[:, true_id].cpu().numpy()
        
        # Anomaly score is 1 - prob(true_id)
        # Average across all frames in the file
        anomaly_score = np.mean(1.0 - prob_true_id)
        
        sota_anomaly_scores.append(anomaly_score)
        sota_true_labels.append(y_test[i])
        sota_machine_ids.append(ids_test[i])

sota_anomaly_scores = np.array(sota_anomaly_scores)
sota_true_labels = np.array(sota_true_labels)
sota_machine_ids = np.array(sota_machine_ids)

print("\\n--- SOTA Evaluation Results (AUC) ---")
for m_id in np.unique(sota_machine_ids):
    mask = (sota_machine_ids == m_id)
    y_true_machine = sota_true_labels[mask]
    y_scores_machine = sota_anomaly_scores[mask]
    
    auc = roc_auc_score(y_true_machine, y_scores_machine)
    print(f"Machine ID {m_id}: AUC = {auc:.4f}")

overall_auc = roc_auc_score(sota_true_labels, sota_anomaly_scores)
print(f"\\nOverall SOTA AUC: {overall_auc:.4f}")
"""
nb.cells.append(nbf.v4.new_code_cell(code_content_4))

# Also add a markdown cell about the methodological flaw of validation split
md_flaw = """## 7. Methodological Note: Validation Set Flaw
It is important to note a slight methodological flaw in the original validation split in Phase 2. The `train_test_split` took 10% of the training data as a validation set. However, since the training data provided by the DCASE challenge **only contains normal sounds**, the validation set is completely devoid of anomalous samples. 

While this is perfectly fine for early stopping or tracking training convergence, **it is impossible to tune an anomaly decision threshold** (e.g., finding a threshold to maximize F1-score) because we cannot calculate False Negatives or True Positives without anomalous data. In a real-world scenario, we would need to inject a small amount of synthetically generated anomalies or a tiny set of collected true anomalies into the validation set to properly calibrate the threshold before deploying."""
nb.cells.append(nbf.v4.new_markdown_cell(md_flaw))

with open(nb_file, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("SOTA Succeeded")
