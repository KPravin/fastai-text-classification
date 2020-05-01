#if canImport(PythonKit)
    import PythonKit
#else
    import Python
#endif

let plt = Python.import("matplotlib.pyplot")

func draw_plot(ys: PythonObject, xs: PythonObject?, x_label: String, y_label: String, title: String, xscale: String? = nil, figsize: Array<Int> = [25, 10]){
  plt.figure(figsize: figsize)

  if let xs = xs{
    plt.plot(xs, ys)
  } else {
    plt.plot(ys)
  }

  if let xscale = xscale{
    plt.xscale(xscale)
  }

  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.title(title)

  plt.show()
}