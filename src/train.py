from torch.utils.data import DataLoader

import logging
import time

import torch.nn as nn
import torch.nn.functional as F
from captum.attr import DeepLift
from captum.attr import IntegratedGradients
from captum.attr import LRP
from captum.attr import LayerLRP
from captum.attr import NoiseTunnel
from captum.attr import Saliency
from captum.attr import visualization as viz
from torch.utils.data import DataLoader

from src.mlp import MLP
from src.toy_colors_dataset import generate_pytorch_dataset
from src.utils import *
from src.visual import *



def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for (x, y) in iterator:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred = model(x)

        loss = criterion(y_pred, y)

        acc = calculate_accuracy(y_pred, y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def get_predictions(model, iterator, device):
    model.eval()

    images = []
    labels = []
    probs = []

    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)

            y_pred = model(x)

            y_prob = F.softmax(y_pred, dim=-1)
            top_pred = y_prob.argmax(1, keepdim=True)

            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_prob.cpu())

    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)

    return images, labels, probs


def train_loop(args, model, train_iterator, dev_iterator, optimizer, criterion, device):
    # Setup optimizer & LR

    if args.getboolean('Train', 'do_train'):
        train_losses = []
        train_accs = []
        valid_losses = []
        valid_accs = []
        best_valid_loss = float('inf')
        for epoch in range(args.getint('Train', 'epochs')):
            start_time = time.monotonic()

            train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
            valid_loss, valid_acc = evaluate(model, dev_iterator, criterion, device)

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            valid_losses.append(valid_loss)
            valid_accs.append(valid_acc)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), args.get('Model', 'models_main_dir') +
                           sorted(os.listdir(args.get('Model', 'models_main_dir')))[-1] + '/' + args.get('Model',
                                                                                                         'model_name') + '.pt')

            end_time = time.monotonic()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    if args.getboolean('Visual', 'plot_train_dev_losses'):
        plt.plot(train_losses, label="Training loss")
        plt.plot(valid_losses, label="Validation loss")
        plt.legend()

    if args.getboolean('Visual', 'plot_train_dev_accuracies'):
        plt.plot(train_accs, label="Training accuracy")
        plt.plot(valid_accs, label="Test accuracy")
        plt.legend()


def test_loop(args, model, test_iterator, criterion, device):
    if args.get('Model', 'cached_model_dir') != "None":
        try:
            if os.path.exists(args.get('Model', 'cached_model_dir')):
                model.load_state_dict(torch.load(args.get('Model', 'cached_model_dir')))
        except Exception:
            model.load_state_dict(torch.load(
                args.get('Model', 'models_main_dir') + sorted(os.listdir(args.get('Model', 'models_main_dir')))[
                    -1] + '/' + args.get('Model', 'model_name') + '.pt'))
    else:
        model.load_state_dict(torch.load(
            args.get('Model', 'models_main_dir') + sorted(os.listdir(args.get('Model', 'models_main_dir')))[
                -1] + '/' + args.get('Model', 'model_name') + '.pt'))

    images, labels, probs = get_predictions(model, test_iterator, device)
    pred_labels = torch.argmax(probs, 1)

    if args.getboolean('Visual', 'plot_confusion_matrix'):
        plot_confusion_matrix(labels, pred_labels)

    corrects = torch.eq(labels, pred_labels)

    incorrect_examples = []

    class1 = []
    class2 = []
    for image, label, prob, correct in zip(images, labels, probs, corrects):
        if label == 0:
            class1.append((image, label, prob))
        if label == 1:
            class2.append((image, label, prob))

        if not correct:
            incorrect_examples.append((image, label, prob))

    incorrect_examples.sort(reverse=True, key=lambda x: torch.max(x[2], dim=0).values)

    return class1, class2, incorrect_examples


def main():
    train_dataset, dev_dataset, test_dataset = generate_pytorch_dataset(
        cachefile='../Data/toy_colors/toy-colors_5_5_3.npz')

    train_params = {'batch_size': 128,
                    'shuffle': True,
                    'num_workers': 6}
    dev_params = {'batch_size': 128,
                  'shuffle': True,
                  'num_workers': 6}

    test_params = {'batch_size': 128,
                   'shuffle': False,
                   'num_workers': 6}

    train_iterator = DataLoader(train_dataset, **train_params)
    dev_iterator = DataLoader(dev_dataset, **dev_params)
    test_iterator = DataLoader(test_dataset, **test_params)

    # dataiter = iter(train_iterator)
    # images1, labels1 = dataiter.next()
    # images2, labels2 = dataiter.next()
    # images3, labels3 = dataiter.next()

    model = MLP(5 * 5 * 3, 2)

    # print(model)
    # from torchviz import make_dot
    # mages1, labels1 = iter(train_iterator).next()
    # yhat = model(images1)
    # make_dot(yhat, params=dict(list(model.named_parameters()))).render("mlp_torchviz", format="png")
    # import hiddenlayer as hl
    # transforms = [hl.transforms.Prune('Constant')]  # Removes Constant nodes from graph.
    # graph = hl.build_graph(model, images1, transforms=transforms)
    # graph.theme = hl.graph.THEMES['blue'].copy()
    # graph.save('mlp_hiddenlayer', format='png')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)

    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []

    do_train = False

    if do_train:
        best_valid_loss = float('inf')

        for epoch in range(5):
            start_time = time.monotonic()

            train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
            valid_loss, valid_acc = evaluate(model, dev_iterator, criterion, device)

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            valid_losses.append(valid_loss)
            valid_accs.append(valid_acc)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), 'baseline_model.pt')

            end_time = time.monotonic()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    model.load_state_dict(torch.load('baseline_model.pt'))
    test_loss, test_acc = evaluate(model, test_iterator, criterion, device)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')

    # plt.plot(train_losses, label="Training loss")
    # plt.plot(valid_losses, label="Validation loss")
    # plt.legend()

    # plt.plot(train_accs, label="Training accuracy")
    # plt.plot(valid_accs, label="Test accuracy")
    # plt.legend()

    images, labels, probs = get_predictions(model, test_iterator, device)
    pred_labels = torch.argmax(probs, 1)

    # plot_confusion_matrix(labels, pred_labels)

    corrects = torch.eq(labels, pred_labels)

    incorrect_examples = []

    class1 = []
    class2 = []
    for image, label, prob, correct in zip(images, labels, probs, corrects):
        if label == 0:
            class1.append((image, label, prob))
        if label == 1:
            class2.append((image, label, prob))

        if not correct:
            incorrect_examples.append((image, label, prob))

    incorrect_examples.sort(reverse=True, key=lambda x: torch.max(x[2], dim=0).values)

    entit = class2[666]
    output_score = entit[2][0].item()
    img = entit[0]
    input = img.unsqueeze(0)
    outputs = model(input)
    _, predicted = torch.max(outputs, 1)

    input.requires_grad = True

    def attribute_image_features(algorithm, input, target, **kwargs):
        model.zero_grad()
        tensor_attributions = algorithm.attribute(input,
                                                  target=target,
                                                  **kwargs
                                                  )

        return tensor_attributions

    saliency = Saliency(model)
    grads = saliency.attribute(input, target=0)
    grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))

    ig = IntegratedGradients(model)
    attr_ig, delta = attribute_image_features(ig, input, baselines=input * 0, return_convergence_delta=True)
    attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
    print('Approximation delta: ', abs(delta))

    ig = IntegratedGradients(model)
    nt = NoiseTunnel(ig)
    attr_ig_nt = attribute_image_features(nt, input, baselines=input * 0, nt_type='smoothgrad_sq',
                                          nt_samples=100, stdevs=0.2)
    attr_ig_nt = np.transpose(attr_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

    dl = DeepLift(model)
    attr_dl = attribute_image_features(dl, input, baselines=input * 0)
    attr_dl = np.transpose(attr_dl.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

    lrp = LRP(model)
    attr_lrp = attribute_image_features(lrp, input)
    attr_lrp = np.transpose((attr_lrp * output_score).squeeze(0).cpu().detach().numpy(), (1, 2, 0))

    layerlrp = LayerLRP(model, [model.layers[0], model.layers[2], model.layers[4]])
    attrs = layerlrp.attribute(input, 1, attribute_to_layer_input=True)
    ll = np.transpose(attrs[0].view(1, 3, 5, 5).squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    l1 = attrs[0].cpu().detach().numpy()
    print(l1)
    l2 = attrs[1].cpu().detach().numpy()
    print(l2)
    l3 = attrs[2].cpu().detach().numpy()
    print(l3)

    # weights = model.layers[0].weight.data

    print('Original Image')
    print('Predicted:', "class 1",
          ' Probability:', torch.max(F.softmax(outputs, 1)).item())

    original_image = np.transpose((img.cpu().detach().numpy()), (1, 2, 0))

    _ = viz.visualize_image_attr(None, original_image,
                                 method="original_image", title="Original Image")

    # _ = viz.visualize_image_attr(attr_lrp, original_image, method="blended_heat_map", sign="positive", show_colorbar=True,
    #                             title="Overlayed LRP")

    _ = viz.visualize_image_attr(ll, original_image, method="heat_map", sign="all", show_colorbar=True, use_pyplot=True,
                                 title="Layer-LRP-heat_map (positive & negative attribution)")
    _ = viz.visualize_image_attr(ll, original_image, method="heat_map", sign="positive", show_colorbar=True,
                                 use_pyplot=True, title="Layer-LRP-heat_map (positive attribution)")
    _ = viz.visualize_image_attr(ll, original_image, method="heat_map", sign="negative", show_colorbar=True,
                                 use_pyplot=True, title="Layer-LRP-heat_map (negative attribution)")

    # _ = viz.visualize_image_attr(ll, original_image, method="blended_heat_map", sign="positive", show_colorbar=True,
    #                             title="Layer-LRP-blended_heat_map")

    # a = input.view(input.size(0), -1)
    # b = a.view(a.size(0), 3, 5, 5)
    # recover_img = np.transpose(b.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    # _ = viz.visualize_image_attr(None, recover_img,
    #                             method="original_image", title="rec Original Image")

    """
    _ = viz.visualize_image_attr(None, original_image,
                                 method="original_image", title="Original Image")

    _ = viz.visualize_image_attr(grads, original_image, method="blended_heat_map", sign="absolute_value",
                                 show_colorbar=True, title="Overlayed Gradient Magnitudes")

    _ = viz.visualize_image_attr(attr_ig, original_image, method="blended_heat_map", sign="all",
                                 show_colorbar=True, title="Overlayed Integrated Gradients")

    _ = viz.visualize_image_attr(attr_ig_nt, original_image, method="blended_heat_map", sign="absolute_value",
                                 outlier_perc=10, show_colorbar=True,
                                 title="Overlayed Integrated Gradients \n with SmoothGrad Squared")

    _ = viz.visualize_image_attr(attr_dl, original_image, method="blended_heat_map", sign="all", show_colorbar=True,
                                 title="Overlayed DeepLift")



    """

    # _ = viz.visualize_image_attr(ll, original_image, method="blended_heat_map", sign="all", show_colorbar=True,
    #                             title="Layer-LRP-blended_heat_map")

    # plot_most_incorrect(incorrect_examples, 30)
    # plot_most_incorrect(class1, 30)
    # plot_most_incorrect(class2, 30)

    N_WEIGHTS = 25

    weights = model.layers[0].weight.data
    # weightss = model.layers[0].weight.data.view(1, 64)

    # plot_weights(weights, N_WEIGHTS)
    plt.show()


if __name__ == '__main__':
    main()
