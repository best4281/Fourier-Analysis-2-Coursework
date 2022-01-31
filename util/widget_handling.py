import enum
from matplotlib import pyplot as plt
from matplotlib.widgets import CheckButtons, Button, Slider, RadioButtons, RangeSlider


class WidgetType(enum.IntEnum):
    """
    Enum for the different types of widgets that will be used in the interactive filter.
    """

    BUTTON = 1
    SLIDER = 2
    RANGESLIDER = 3
    CHECKBUTTONS = 4
    RADIOBUTTONS = 5
    OTHER = 0


class InteractionWidgets:
    def __init__(self, x, y, width, length, widget_type: WidgetType, callback=None, **kwargs):
        """
        Class for the interactive widgets in matplotlib figure.
        :param x: x-coordinate of the widget
        :param y: y-coordinate of the widget
        :param width: width of the widget
        :param length: length of the widget
        :param widget_type: type of the widget
        :param callback: callback function to be called when the widget is interacted with
        :param other_type: if the widget is of type OTHER, this should be the type of the widget
        :param kwargs: additional arguments for the widget constructor, depending on the widget type in matplotlib library
        """
        self.x = x
        self.y = y
        self.width = width
        self.length = length
        self.widget_type = widget_type
        self.ax = plt.axes([x, y, width, length])

        if callback is None:
            self.callback = lambda event: None
        else:
            self.callback = callback

        if self.widget_type == WidgetType.BUTTON:
            self.widget = Button(self.ax, **kwargs)
        elif self.widget_type == WidgetType.SLIDER:
            self.widget = Slider(self.ax, **kwargs)
        elif self.widget_type == WidgetType.RANGESLIDER:
            self.widget = RangeSlider(self.ax, **kwargs)
        elif self.widget_type == WidgetType.CHECKBUTTONS:
            self.widget = CheckButtons(self.ax, **kwargs)
        elif self.widget_type == WidgetType.RADIOBUTTONS:
            self.widget = RadioButtons(self.ax, **kwargs)
        elif self.widget_type == WidgetType.OTHER and "other_type" in kwargs:
            self.widget = kwargs["other_type"](self.ax, **kwargs)
            self.other_type = kwargs["other_type"]
            if callback() is not None:
                pass
                # TODO: add callback
        else:
            raise ValueError("Widget type is not specified")
        self.connect_callback()

    def disconnect_callback(self):
        self.widget.disconnect(self.cid)

    def connect_callback(self):
        if self.callback is None:
            return
        if self.widget_type == WidgetType.BUTTON:
            self.cid = self.widget.on_clicked(self.callback)
        elif self.widget_type == WidgetType.SLIDER or self.widget_type == WidgetType.RANGESLIDER:
            self.cid = self.widget.on_changed(self.callback)
        elif self.widget_type == WidgetType.CHECKBUTTONS or self.widget_type == WidgetType.RADIOBUTTONS:
            self.cid = self.widget.on_clicked(self.callback)
        else:
            pass
            # TODO: add callback
