#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def plot_activation_function(function, wider=False, title_kwargs=None):
    decorator_return = function
    if isinstance(function, type):
        name = function.__name__
        if issubclass(function, torch.autograd.Function):  # autograd Function
            function = function.apply
        elif issubclass(function, nn.Module):  # uninitialized class of Module
            function = function()
        else:
            assert False
    else:
        name = type(function).__name__
        if isinstance(function, torch.autograd.Function):  # autograd Function
            function = function.apply
        elif not isinstance(function, nn.Module):  # assert function, method, or built-in function
            assert callable(function)
            name = function.__name__
    tick_major = np.arange(-6., 7., 2) if wider else np.arange(-4., 5., 1)
    tick_minor = np.arange(-7., 8., 1) if wider else np.arange(-4., 5., 1)
    x = torch.arange(tick_minor.min(), tick_minor.max(), 1e-3).requires_grad_()
    y = function(x)
    dydx = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    d2ydx2 = torch.autograd.grad(dydx.sum(), x)[0]
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
    for ax, data, color, title_format in zip(
        axes, (y, dydx, d2ydx2), "rbg", ("{} Function", "Derivative of {}", "Second Derivative of {}")
    ):
        ax.axvline(0., c='k', linewidth=1., alpha=0.7)
        ax.axhline(0., c='k', linewidth=1., alpha=0.7)
        ax.plot(x.detach().cpu().numpy(), data.detach().cpu().numpy(), c=color)
        ax.set_xticks(tick_major)
        ax.set_yticks(tick_major)
        ax.set_xticks(tick_minor, minor=True)
        ax.set_yticks(tick_minor, minor=True)
        ax.grid(which="major",alpha=0.5)
        ax.grid(which="minor",alpha=0.5)
        if title_kwargs:
            title_format += " (" + ", ".join("%s=%s" % (k, v) for k, v in title_kwargs.items()) + ")"
        ax.set_title(title_format.format(name.replace("Function", "").replace("function", "")))
    plt.show()
    return decorator_return


def plot_activation_module(draw_wider=False, **initkwargs):
    def decorator(klass):
        return plot_activation_function(klass(**initkwargs), draw_wider, initkwargs)
    if isinstance(draw_wider, type) and issubclass(draw_wider, nn.Module):
        klass = draw_wider
        draw_wider = False
        return decorator(klass)
    assert isinstance(draw_wider, bool)
    return decorator
