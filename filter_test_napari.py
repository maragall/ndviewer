import napari

viewer = napari.Viewer()

def _myfilter(row, parent):
    return "<hidden>" not in viewer.layers[row].name

viewer.window.qt_viewer.layers.model().filterAcceptsRow = _myfilter

viewer.add_points(None, name='A')
viewer.add_points(None, name='B <hidden>')
viewer.add_points(None, name='C')

napari.run()