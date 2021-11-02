from captum.attr import LayerLRP
from captum.attr import visualization as viz

from src.utils import *
from src.visual import *



def attribute_image_features(model, algorithm, input, target, **kwargs):
    model.zero_grad()
    tensor_attributions = algorithm.attribute(input,
                                              target=target,
                                              **kwargs
                                              )

    return tensor_attributions


def explain(method, model, data_point):
    input = data_point[0].unsqueeze(0)
    input.requires_grad = True

    if method == "LayerLRP":
        layerlrp = LayerLRP(model, [model.layers[0], model.layers[2], model.layers[4]])
        attrs_layerlrp = attribute_image_features(model, layerlrp, input, data_point[1].item(),
                                                  attribute_to_layer_input=True)
        ll = np.transpose(attrs_layerlrp[0].view(1, 3, 5, 5).squeeze(0).cpu().detach().numpy(), (1, 2, 0))

        original_image = np.transpose((data_point[0].cpu().detach().numpy()), (1, 2, 0))

        fig1, ax1 = plt.subplots(2, 2, figsize=(15, 10))

        _ = viz.visualize_image_attr(None, original_image, method="original_image", use_pyplot=False,
                                     title="Original Image: Class " + str(data_point[1].item()),
                                     plt_fig_axis=(fig1, ax1[0, 0]))

        _ = viz.visualize_image_attr(ll, original_image, method="heat_map", sign="all", show_colorbar=True,
                                     use_pyplot=False,
                                     title="Layer-LRP-heat_map (positive & negative attribution)",
                                     plt_fig_axis=(fig1, ax1[0, 1]))
        _ = viz.visualize_image_attr(ll, original_image, method="heat_map", sign="positive", show_colorbar=True,
                                     use_pyplot=False, title="Layer-LRP-heat_map (positive attribution)",
                                     plt_fig_axis=(fig1, ax1[1, 0]))
        _ = viz.visualize_image_attr(ll, original_image, method="heat_map", sign="negative", show_colorbar=True,
                                     use_pyplot=False, title="Layer-LRP-heat_map (negative attribution)",
                                     plt_fig_axis=(fig1, ax1[1, 1]))

        return fig1, ax1
