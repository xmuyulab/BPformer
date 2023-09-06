import sys
import torch
from tqdm import tqdm  # 导入tqdm库

def train_one_epoch(now_epoch, all_epoch, model, optimizer, data_loader, num, criterion, device):
    model.train()
    optimizer.zero_grad()

    print("train: {}".format(num))
    accurate = 0
    loss_sum = 0
    running_loss = 0.0
    with tqdm(data_loader, desc=f"Epoch {now_epoch+1}/{all_epoch}", ncols=80) as pbar:
        for x_mRNA, y_label in pbar:
            x_mRNA = x_mRNA.to(device)
            y_label = y_label.to(device)
            pred = model(x_mRNA)
            loss = criterion(pred, y_label)
            _, pred = torch.max(pred, 1)
            accurate += (pred == y_label).sum()

            loss_sum += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            # 更新进度条信息
            pbar.set_postfix({"Loss": running_loss / (pbar.n + 1)})
            pbar.update()
    
    return loss_sum/num, accurate/num
 
# @torch.no_grad()
def train_evaluate(model, data_loader, num, criterion, device):
    accurate = 0
    loss_sum = 0
    true_label = []
    pred_label = []
    print("test: {}".format(num))

    model.eval() 
    
    with torch.no_grad():
        for x_mRNA, y_label in data_loader:
            x_mRNA = x_mRNA.to(device)
            y_label = y_label.to(device)
            pred = model(x_mRNA)
            loss = criterion(pred, y_label)
            _, pred = torch.max(pred, 1)
            true_label.extend(y_label.tolist())
            pred_label.extend(pred.tolist())
            accurate += (pred == y_label).sum()
            loss_sum += loss.item()

    return loss_sum/num, accurate/num, true_label, pred_label

# @torch.no_grad()
def evaluate(model, data_loader, num, criterion, device):
    accurate = 0
    loss_sum = 0

    print("test: {}".format(num))

    model.eval()
    
    pred_label = []
    total_num = 0
  
    with torch.no_grad():
        for x_mRNA, y_label in data_loader:
            x_mRNA = x_mRNA.to(device)
            y_label = y_label.to(device)

            pred = model(x_mRNA)

            # print(pred.shape)

            loss = criterion(pred, y_label)
            _, pred = torch.max(pred, 1)

            pred_label.extend(pred.tolist())

            accurate += sum(pred == y_label)

            loss_sum += loss.item()

    return loss_sum/num, accurate/num, pred_label


# @torch.no_grad()
def evaluate_zhongshan(model, data_loader, num, criterion, device):
    accurate = 0
    loss_sum = 0

    print("test: {}".format(num))

    model.eval()
    
    pred_label = []

    pred_label_top5 = []
    pred_label_top3 = []
  
    with torch.no_grad():
        for x_mRNA, y_label in data_loader:
            x_mRNA = x_mRNA.to(device)
            y_label = y_label.to(device)

            pred = model(x_mRNA)

            # 获取前 k 个最大值及其索引
            top_values_5, top_indices_5 = torch.topk(pred, 5)
            top_values_3, top_indices_3 = torch.topk(pred, 3)

            # print(top_values)     # 输出前 k 个最大值
            # print(top_indices)    # 输出前 k 个最大值对应的索引
            pred_label_top5.append(top_indices_5.tolist())
            pred_label_top3.append(top_indices_3.tolist())


            loss = criterion(pred, y_label)
            _, pred = torch.max(pred, 1)

            pred_label.extend(pred.tolist())

            accurate += sum(pred == y_label)

            loss_sum += loss.item()
    
    return loss_sum/num, accurate/num, pred_label, pred_label_top5, pred_label_top3