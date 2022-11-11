import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

def testGCN(model, test_flow):
    test_prediction = model.predict(test_flow)
    analyze_performance(test_prediction, test_flow)
    create_histogram(test_prediction, test_flow)
    create_confusion_matrix(test_prediction, test_flow)

def analyze_performance(test_flow, test_prediction):
    flto = test_flow.targets#new_test_pre.flatten() # Ground truth set
    fltp = test_prediction.flatten() # prediction outputs set (test_cal_pred)

    auc = roc_auc_score(flto, fltp)
    fpr, tpr, ths = roc_curve(flto, fltp)
    opt = np.argmax(tpr - fpr)
    th = ths[opt] # This is also important for calculating the optimal threshold for your classifier
    cm = confusion_matrix(flto>0.5, fltp>th, normalize='all')
    tn, fp, fn, tp = cm.ravel()
    fn = fn.astype("float64")
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    acc = (tp + tn)*100 / (tp + tn + fp + fn)
    j = tp / (tp + fn) + tn / (tn + fp) - 1
    p = tp / (fp + tp)
    r = tp / (fn + tp)
    f1 = 2*p*r / (p + r)
    with open(f"performance.txt", "w") as file:
        def print_(*args, **kwargs): print(*args, **kwargs); print(*args, **kwargs, file=file)
        #model.summary(print_fn=lambda x: print_(x))
        print_()
        # print_(f"Dropout: {dp1} | {dp2}")
        print_()
        #print_(f"ML Test Loss: {results[0]:.9f}")
        #print_(f"ML Test Acc: {results[1]*100:.3f}%")
        print_(f"Threshold: {th:.5f}")
        print_(f"True Positive: {tp:.0f}")
        print_(f"False Negative: {fn:.0f}")
        print_(f"True Negative: {tn:.0f}")
        print_(f"False Positive: {fp:.0f}")
        print_(f"J   = {j}")
        print_(f"MCC = {mcc}")
        print_(f"AUC = {auc}")
        print_(f"Accuracy = {acc:.4f}%")
        print_(f"Precision = {p:.4f}")
        print_(f"Recall = {r:.4f}")
        print_(f"F1 Score = {f1:.4f}")

def create_histogram(test_prediction, test_flow):
    fig = plt.figure(figsize = (12,8), facecolor=('xkcd:white'))
    plt.hist(test_prediction[np.where(test_flow.targets[np.sort(test_flow.indices)] > 0)], color="green", label="Positive Samples", alpha=0.5, bins=50)
    plt.hist(test_prediction[np.where(test_flow.targets[np.sort(test_flow.indices)] == 0)], color="red", label="Negative Samples", alpha=0.5, bins=50)
    plt.legend(loc="upper left", fontsize=10)
    plt.xlabel('Weight Predicted')
    plt.ylabel('# of Links')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title("Distribution Histogram of Predictions", fontsize = 18)
    plt.savefig('hist.png')
    plt.show()

def create_confusion_matrix(cm):
    font = {'family' : 'normal',
    'weight' : 'bold',
    'size'   : 22}
    plt.rc('font', **font)
    # dfcm = pd.DataFrame(cm, index = [i for i in "01"],
    #                   columns = [i for i in "01"])
    plt.figure(figsize = (10,7), facecolor=('xkcd:white'))
    mi, ma = cm.min(), cm.max()
    sn.heatmap(cm, annot=True,  fmt='.4f', norm=LogNorm(vmin=mi, vmax=ma), cmap="Blues")
    #plt.savefig(f'{directory}/cm.png')
    plt.title('Confusion Matrix', fontsize=18)
    plt.xlabel(f'Predicted Label\n\nAccuracy={acc:0.4f}% MCC={mcc:0.4f}\nPrecision={p:0.4f}  Recall={r:0.4f}')
    plt.ylabel('True Label')
    plt.show(block=False)