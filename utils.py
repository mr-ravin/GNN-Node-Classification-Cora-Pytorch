import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd 

def save_plot(train_loss_list=[], filter_bucket=100):
    plt.figure()
    red_patch = mpatches.Patch(color='red', label='Train Loss')
    x_axis_data = list(range(1,len(train_loss_list)+1))
    x_axis_data = [x * filter_bucket for x in x_axis_data]
    sns.lineplot(x=x_axis_data, y=train_loss_list, color='red', alpha=0.75)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GNN Training Analysis")
    plt.legend(handles=[red_patch], loc='upper right')
    plt.savefig('./result/training_analysis.png')
    print("Saved ./result/training_analysis.png")

def filtered_result(value_list, filter_bucket=20):
    result = []
    len_value_list = len(value_list)
    collect_at_each = int(len_value_list/filter_bucket)
    for idx in range(1, len_value_list+1):
        if idx % collect_at_each == 0:
            result.append(value_list[idx-1])
    return result, collect_at_each