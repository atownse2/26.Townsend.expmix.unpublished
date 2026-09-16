import ROOT
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from typing import Dict
from array import array

from .fitting import fit

random_string = lambda: ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=10))


default_colors = [
    "#D55E00",  # Vermillion
    "#CC79A7",  # Reddish Purple
    "#009E73",  # Bluish Green
    "#F0E442",  # Yellow
    "#0072B2",  # Blue
    "#56B4E9",  # Sky Blue
    "#E69F00",  # Orange
    # "#000000"   # Black
]


default_root_colors = [ROOT.TColor.GetColor(c) for c in default_colors]

def roo_hist_to_th1(roo_hist, name):
    n_points = roo_hist.GetN()
    x_values = list(roo_hist.GetX())
    y_values = list(roo_hist.GetY())

    if n_points == 0:
        return ROOT.TH1D(name, "", 1, 0.0, 1.0)

    if n_points == 1:
        width = 1.0
        edges = [x_values[0] - width / 2.0, x_values[0] + width / 2.0]
    else:
        edges = [x_values[0] - (x_values[1] - x_values[0]) / 2.0]
        for i in range(n_points - 1):
            edges.append((x_values[i] + x_values[i + 1]) / 2.0)
        edges.append(x_values[-1] + (x_values[-1] - x_values[-2]) / 2.0)

    hist = ROOT.TH1D(name, "", len(edges) - 1, array('d', edges))
    hist.SetDirectory(0)
    for i, value in enumerate(y_values):
        hist.SetBinContent(i + 1, value)
    return hist

def plot_fits(
    data, x,
    models,
    labels,
    title=None,
    logx=False,
    x_label=None,
    nbins=None,
    plot_range=None,
    pull_range=(-4, 4),
    binning=None,
    colors = default_root_colors,
    linestyles=None,
    y_min=1.1e-1,
    markersize=1.5,
    linewidth=2,
    title_size=0.12, label_size=0.09,
    logy=True,
    legend_bounds=(0.55, 0.56, 0.95, 0.85),
    legend_columns=2,
    legend_text_size=0.064,
    legend_margin=0.18,
    pull_fill_style=1001,
    pull_fill_alpha=0.3,
    save_as=None
    ):

    c = ROOT.TCanvas(random_string(), "canvas", 1600, 800)
    c.cd()

    # Set binning
    if nbins is not None:
        x_min = x.getMin()
        x_max = x.getMax()
        if plot_range is not None:
            x_min, x_max = plot_range
        binning = ROOT.RooBinning(nbins, x_min, x_max)
        x.setBinning(binning)

    # Frames
    main_frame = x.frame()

    if title is not None:
        main_frame.SetTitle(title)
        main_frame.SetTitleSize(0.08)
    else:
        main_frame.SetTitle("")

    main_frame.GetXaxis().SetTitle(x_label if x_label is not None else x.GetTitle())
    main_frame.GetYaxis().SetTitle("Events / bin")
    main_frame.GetYaxis().CenterTitle(True)
    main_frame.GetYaxis().SetTitleSize(title_size)
    main_frame.GetYaxis().SetTitleOffset(0.42)
    main_frame.GetYaxis().SetLabelSize(label_size)

    pull_frame = x.frame()
    pull_frame.SetTitle("")  # Remove title
    pull_frame.GetYaxis().SetTitleSize(title_size)
    pull_frame.GetYaxis().SetTitleOffset(0.37)
    pull_frame.GetYaxis().SetTitle("#frac{data - fit}{#sigma_{data}}")
    pull_frame.GetYaxis().CenterTitle(True)
    pull_frame.GetYaxis().SetLabelSize(label_size)
    pull_frame.GetYaxis().SetNdivisions(104, False)

    # Add dashed line at y=0
    min_x = x.getMin()
    max_x = x.getMax()
    if plot_range is not None:
        min_x, max_x = plot_range
    line = ROOT.TLine(min_x, 0, max_x, 0)
    line.SetLineStyle(ROOT.kDashed)
    line.SetLineColor(ROOT.kBlack)
    pull_frame.addObject(line)

    pull_frame.GetXaxis().SetTitleSize(title_size + 0.02)
    pull_frame.GetXaxis().SetTitleOffset(1.1)
    pull_frame.GetXaxis().CenterTitle(True)
    pull_frame.GetXaxis().SetLabelSize(label_size + 0.01)
    pull_frame.GetXaxis().SetTitle(x_label if x_label is not None else x.GetTitle())

    
    if plot_range is not None:
        main_frame.GetXaxis().SetRangeUser(*plot_range)
        pull_frame.GetXaxis().SetRangeUser(*plot_range)

    if logx:
        # Improve readability on log-x by drawing labels on intermediate ticks.
        main_frame.GetXaxis().SetMoreLogLabels(True)
        pull_frame.GetXaxis().SetMoreLogLabels(True)
        main_frame.GetXaxis().SetNoExponent(True)
        pull_frame.GetXaxis().SetNoExponent(True)

    # Plot data and fits
    data_reference_name = "plot_data_reference"
    data_name = "plot_data"
    data.plotOn(
        main_frame,
        ROOT.RooFit.Name(data_reference_name),
        ROOT.RooFit.MarkerSize(0),
        ROOT.RooFit.LineColor(0),
    )

    curve_specs = []
    pull_specs = []
    for i, model in enumerate(models):
        if hasattr(model, "pdf"):
            pdf = model.pdf
        else:
            pdf = model

        model_label = labels[i]
        if colors is not None:
            if len(colors) == 0:
                raise ValueError("colors must not be empty when provided")
            model_color = colors[i % len(colors)]
        else:
            model_color = colors[i % len(colors)]
        if isinstance(model_color, str):
            model_color = ROOT.TColor.GetColor(model_color)

        if linestyles is not None:
            if len(linestyles) == 0:
                raise ValueError("linestyles must not be empty when provided")
            line_style = linestyles[i % len(linestyles)]
            if isinstance(line_style, str):
                line_style_lookup = {
                    "solid": 1,
                    "dashed": 2,
                    "dotted": 3,
                    "dashdotted": 4,
                    "dashdot": 4,
                }
                line_style = line_style_lookup.get(line_style.strip().lower())
                if line_style is None:
                    raise ValueError(f"Unsupported line style: {linestyles[i % len(linestyles)]}")
        else:
            line_style = ROOT.kSolid

        curve_name = f"plot_curve_{i}"

        pdf.plotOn(
            main_frame,
            ROOT.RooFit.Precision(1e-5),
            ROOT.RooFit.LineColor(model_color),
            ROOT.RooFit.LineStyle(line_style),
            ROOT.RooFit.LineWidth(int(linewidth)),
            ROOT.RooFit.Name(curve_name),
            ROOT.RooFit.DrawOption("L"),  # Use "L" for line only
        )
        curve_specs.append((curve_name, model_label, model_color))

    data.plotOn(
        main_frame,
        ROOT.RooFit.Name(data_name),
        ROOT.RooFit.MarkerSize(markersize),
    )

    for curve_name, model_label, model_color in curve_specs:
        pull_hist = main_frame.pullHist(data_reference_name, curve_name)
        pull_hist.SetLineColor(model_color)
        pull_hist.SetMarkerColor(model_color)
        pull_hist.SetMarkerSize(markersize)
        pull_hist.SetLineWidth(int(linewidth))
        pull_specs.append((pull_hist, model_color, pull_fill_style))

    # Plot the histograms
    main_pad = ROOT.TPad("main_pad", "Main Pad", 0, 0.5, 1, 1)
    pull_pad = ROOT.TPad("pull_pad", "Pull Pad", 0, 0, 1, 0.5)

    if logy:
        main_pad.SetLogy()
    if logx:
        main_pad.SetLogx()
    main_pad.SetTopMargin(0.12)
    main_pad.SetLeftMargin(0.12)
    main_pad.SetRightMargin(0.04)
    main_pad.SetBottomMargin(0)
    main_pad.Draw()

    if logx:
        pull_pad.SetLogx()
    pull_pad.SetLeftMargin(0.12)
    pull_pad.SetRightMargin(0.04)
    pull_pad.SetTopMargin(0)
    pull_pad.SetBottomMargin(0.38)
    pull_pad.Draw()

    legend_x1, legend_y1, legend_x2, legend_y2 = legend_bounds
    legend = ROOT.TLegend(legend_x1, legend_y1, legend_x2, legend_y2)
    legend.SetNColumns(legend_columns)
    legend.SetTextFont(42)
    legend.SetTextSize(legend_text_size)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)  # Transparent legend background
    legend.SetMargin(legend_margin)

    data_obj = main_frame.findObject(data_name)
    if data_obj:
        legend.AddEntry(data_obj, "Data", "lp")

    for curve_name, model_label, model_color in curve_specs:
        curve_obj = main_frame.findObject(curve_name)
        if curve_obj:
            legend.AddEntry(curve_obj, model_label, "l")

    main_pad.cd()
    main_frame.SetMinimum(y_min)
    if logy:
        main_frame.SetMaximum(main_frame.GetMaximum() * 10)
    else:
        main_frame.SetMaximum(main_frame.GetMaximum() * 1.2)
    main_frame.Draw()

    legend.Draw()
    ROOT.SetOwnership(legend, False)

    pull_pad.cd()
    pull_frame.Draw()

    pull_hist_frame = next(
        (prim for prim in pull_pad.GetListOfPrimitives() if isinstance(prim, ROOT.TH1)),
        None,
    )
    if pull_hist_frame is not None:
        pull_hist_frame.SetMinimum(pull_range[0])
        pull_hist_frame.SetMaximum(pull_range[1])
        pull_hist_frame.GetYaxis().SetNdivisions(104, False)
        pull_hist_frame.GetYaxis().SetTitle("#frac{data - fit}{#sigma_{data}}")
        pull_hist_frame.GetYaxis().CenterTitle(True)
        pull_hist_frame.GetYaxis().SetTitleSize(title_size)
        pull_hist_frame.GetYaxis().SetTitleOffset(0.37)
        pull_hist_frame.GetYaxis().SetLabelSize(label_size)
        pull_hist_frame.GetXaxis().SetTitle(x_label if x_label is not None else x.GetTitle())
        pull_hist_frame.GetXaxis().SetTitleSize(title_size + 0.02)
        pull_hist_frame.GetXaxis().SetTitleOffset(1.1)
        pull_hist_frame.GetXaxis().CenterTitle(True)
        pull_hist_frame.GetXaxis().SetLabelSize(label_size + 0.01)

    filled_pull_hists = []
    for i, (pull_hist, model_color, fill_style) in enumerate(pull_specs):
        filled_hist = roo_hist_to_th1(pull_hist, f"pull_fill_{i}_{random_string()}")
        filled_hist.SetLineColor(model_color)
        filled_hist.SetLineWidth(1)
        filled_hist.SetFillColorAlpha(model_color, pull_fill_alpha)
        filled_hist.SetFillStyle(fill_style)
        filled_hist.Draw("HIST SAME")
        filled_pull_hists.append(filled_hist)

    line.Draw("SAME")

    c._pull_filled_hists = filled_pull_hists
    c._pull_line = line

    c.Update()
    c.Draw()

    if save_as is not None:
        c.SaveAs(save_as)

    return c

def plot_correlation_matrix(fit_result, title="Correlation Matrix", save_path=None):
    """
    Plot the correlation matrix from a RooFit result using matplotlib.
    """
    corr_matrix = fit_result.correlationMatrix()
    
    # Get parameter names from the fit result
    param_names = []
    for i in range(fit_result.floatParsFinal().getSize()):
        param = fit_result.floatParsFinal().at(i)
        param_names.append(param.GetName())
    
    # Convert ROOT matrix to numpy array
    n_params = corr_matrix.GetNrows()
    corr_array = np.zeros((n_params, n_params))
    
    for i in range(n_params):
        for j in range(n_params):
            corr_array[i, j] = corr_matrix[i][j]
    
    # Create matplotlib figure
    fac = n_params/4
    fig, ax = plt.subplots(figsize=(4*fac, 3*fac))
    
    # Create heatmap
    im = ax.imshow(corr_array, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    
    # Set ticks and labels
    ax.set_xticks(range(n_params))
    ax.set_yticks(range(n_params))
    ax.set_xticklabels(param_names, rotation=45, ha='right')
    ax.set_yticklabels(param_names)
    
    # Add text annotations
    for i in range(n_params):
        for j in range(n_params):
            value = corr_array[i, j]
            text_color = 'white' if abs(value) > 0.5 else 'black'
            ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                   color=text_color, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', rotation=270, labelpad=15)
    
    # Set title and layout
    ax.set_title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    
    return fig

def plot_information_criteria(
    criteria_by_x,
    x_values=None,
    figsize=(8, 3),
    x_label="Number of Components (k)",
    aic_key="AIC",
    bic_key="BIC",
    aic_label="$\Delta\\text{AIC}$",
    bic_label="$\Delta\\text{BIC}$",
    aic_color="red",
    bic_color="blue",
    bic_marker="x",
    aic_marker_size=100,
    bic_marker_size=100,
    label_size=18,
    tick_label_size=None,
    legend_label_size=None,
    legend_loc="upper center",
    sort_x=True,
    show=True,
    save_as=None,
):
    if isinstance(criteria_by_x, dict):
        items = list(criteria_by_x.items())
    else:
        if x_values is None:
            raise ValueError("x_values is required when criteria_by_x is not a dictionary")
        items = list(zip(x_values, criteria_by_x))

    if not items:
        raise ValueError("No information criteria were provided")

    if sort_x:
        try:
            items = sorted(items, key=lambda item: item[0])
        except TypeError:
            pass

    plot_values = [item[0] for item in items]
    aic_values = []
    bic_values = []
    for x_value, criteria in items:
        if aic_key not in criteria or bic_key not in criteria:
            raise KeyError(
                f"Missing {aic_key} or {bic_key} for {x_value}"
            )
        aic_values.append(criteria[aic_key])
        bic_values.append(criteria[bic_key])

    aic_values = np.array(aic_values)
    bic_values = np.array(bic_values)

    aic_values -= np.min(aic_values)
    bic_values -= np.min(bic_values)

    use_numeric_axis = all(
        isinstance(value, (int, float, np.integer, np.floating))
        for value in plot_values
    )
    if use_numeric_axis:
        x_coords = plot_values
        x_tick_labels = None
    else:
        x_coords = np.arange(len(plot_values))
        x_tick_labels = plot_values

    fig, ax1 = plt.subplots(1, figsize=figsize)

    ax1.scatter(x_coords, aic_values, color=aic_color, label=aic_label, s=aic_marker_size)
    ax1.set_xlabel(x_label, fontsize=label_size)
    ax1.set_ylabel(aic_label, color=aic_color, fontsize=label_size)
    ax1.tick_params(axis='y', labelcolor=aic_color, labelsize=tick_label_size)
    ax1.set_xticks(x_coords)
    ax1.tick_params(axis='x', labelsize=tick_label_size)
    if x_tick_labels is not None:
        ax1.set_xticklabels(x_tick_labels)

    ax2 = ax1.twinx()
    ax2.scatter(
        x_coords,
        bic_values,
        color=bic_color,
        label=bic_label,
        marker=bic_marker,
        s=bic_marker_size,
    )
    ax2.set_ylabel(bic_label, color=bic_color, fontsize=label_size)
    ax2.tick_params(axis='y', labelcolor=bic_color, labelsize=tick_label_size)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc=legend_loc,
        fontsize=legend_label_size,
    )

    fig.tight_layout()
    if show:
        plt.show()

    if save_as is not None:
        fig.savefig(save_as)

    return fig, ax1, ax2

def plot_2D_profile(
        df: pd.DataFrame,
        p1_name, p2_name,
        ax=None, fig=None,
        plot_contours=True,
        worst_case=False,
        logz=False,
        random_restarts=None,
    ):

    X_vals = df[p1_name].unique()
    Y_vals = df[p2_name].unique()

    # Sort
    X_vals.sort()
    Y_vals.sort()

    X_grid, Y_grid = np.meshgrid(X_vals, Y_vals)

    # Compute the minimum NLL over the other dimensions
    Z_grid = np.zeros_like(X_grid)
    for i in range(X_grid.shape[0]):
        for j in range(X_grid.shape[1]):
            mask = (df[p1_name] == X_grid[i, j]) & (df[p2_name] == Y_grid[i, j])
            if not mask.any():
                print(f"Warning: No data point found for ({X_grid[i, j]}, {Y_grid[i, j]})")
            
            if worst_case:
                Z_val = df[mask]['nll'].max()
            else:
                Z_val = df[mask]['nll'].min()

            if np.isnan(Z_val):
                print(f"Warning: No NLL value found for ({X_grid[i, j]}, {Y_grid[i, j]})")
            Z_grid[i, j] = Z_val

    # Compute delta NLL
    delta_nll = Z_grid - np.nanmin(Z_grid)

    # Make plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    # Contour plot (log scale)
    if logz:
        cf = ax.contourf(X_grid, Y_grid, np.log1p(delta_nll + 1e-6), levels=50, cmap='viridis')
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label('log10(ΔNLL)')
    else:
        cf = ax.contourf(X_grid, Y_grid, delta_nll, levels=50, cmap='viridis')
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label('ΔNLL')

    if plot_contours:
        # Add confidence interval contours (on original delta_nll scale)
        sigma_levels = [2.30, 6.18, 11.83]  # 1σ, 2σ, 3σ for 2D
        cl = ax.contour(X_grid, Y_grid, delta_nll, levels=sigma_levels, colors='white', linestyles='dashed')
        ax.clabel(cl, inline=True, fontsize=10)

    # Add star at the maximum likelihood point (minimum NLL)
    # Find the location of the minimum NLL (maximum likelihood)
    min_idx = np.unravel_index(np.nanargmin(Z_grid), Z_grid.shape)
    max_x = X_grid[min_idx]
    max_y = Y_grid[min_idx]
    ax.plot(max_x, max_y, '.', color='red', markersize=1)

    # Add random restart points if provided
    if random_restarts is not None:
        for i, fit_result in enumerate(random_restarts):
            x_val = fit_result['final_pars'][p1_name]
            y_val = fit_result['final_pars'][p2_name]

            if i == len(random_restarts) - 1:
                ax.plot(x_val, y_val, 'x', color='red', markersize=5, label='Best Solution')
            else:
                ax.plot(x_val, y_val, 'x', color='black', markersize=5, label="A Solution")
        ax.legend()

def plot_pair_profiles(
        df: pd.DataFrame,
        pset: Dict[str, tuple],
        plot_contours=True,
        worst_case=False,
        logz=False,
    ):
    params = list(pset.keys())
    n = len(params)
    fig, axes = plt.subplots(n-1, n-1, figsize=(6*(n-1), 6*(n-1)))
    for i, param_x in enumerate(params[:-1]):
        for j, param_y in enumerate(params[1:]):
            if n == 2:
                # Special case for 2 parameters
                ax = axes
            else:
                ax = axes[j, i]
            if i==j+1:
                ax.axis('off')
                continue
            plot_2D_profile(df, param_x, param_y, ax=ax, fig=fig, plot_contours=plot_contours, worst_case=worst_case, logz=logz)
            print(f"Plotting {param_x} vs {param_y} on axes ({j}, {i})")
    
            if j == n-2:
                ax.set_xlabel(param_x)
            if i == 0:
                ax.set_ylabel(param_y)
    
    return fig, axes

