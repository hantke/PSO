
def set_style(useTex=True):
    """Alternative styles for the plots
    
    :param useTex: Use Latex, defaults to True
    :type useTex: bool, optional
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    params = {"text.usetex": useTex,
              
              "axes.labelpad":              10,
              "axes.labelsize":             7,
              "axes.linewidth":             2,
              "axes.labelpad":              10,
              
              "xtick.labelsize":            33,
              "xtick.bottom":               True,
              "xtick.top":                  True,
              "xtick.direction":            'in',
              "xtick.minor.visible":        True,
              "xtick.minor.size":           6,
              "xtick.minor.width":          1,
              "xtick.minor.pad":            4,
              "xtick.major.size":           12,
              "xtick.major.width":          2,
              "xtick.major.pad":            3,

              "ytick.labelsize":            33,
              "ytick.left":                 True,
              "ytick.right":                True,
              "ytick.direction":            'in',
              "ytick.minor.visible":        True,
              "ytick.minor.size":           6,
              "ytick.minor.width":          1,
              "ytick.minor.pad":            4,
              "ytick.major.size":           12,
              "ytick.major.width":          2,
              "ytick.major.pad":            3,

              "figure.figsize":             "10, 10",
              "figure.dpi":                 80,
              "figure.subplot.left":        0.05,
              "figure.subplot.bottom":      0.05,
              "figure.subplot.right":       0.95,
              "figure.subplot.top":         0.95,

              "legend.numpoints":           1,
              "legend.frameon":             False,
              "legend.handletextpad":       0.3,

              "savefig.dpi":                80,

              "font.family":          'serif',

              "path.simplify":              True

              }

    # plt.rc("font", family = "serif")
    plt.rcParams.update(params)
