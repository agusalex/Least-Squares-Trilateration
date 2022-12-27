import matplotlib.pyplot as plt
from random import randint
from easy_trilateration import model
import matplotlib

from easy_trilateration.model import Point


def static(history: [model.Trilateration], actual=None, draw_target_circle=False, title="Multilateration"):
    if actual is None:
        actual = []
    x_values = []
    y_values = []

    sniffers = set()
    to_draw = []
    for i in range(len(history)):
        tri = history[i]
        for sniffer in tri.sniffers:
            sniffers.add(sniffer.center)
        x_values.append(tri.result.center.x)
        y_values.append(tri.result.center.y)
    if draw_target_circle:
        to_draw.append(create_circle(tri.result, target=True))

    for sniff in sniffers:
        to_draw.append(create_point(sniff, color="red"))
    actual_x = []
    actual_y = []

    for x, y in actual:
        actual_x.append(x)
        actual_y.append(y)
    plt.grid()
    plt.title(title)
    plt.ylabel("Y Meters")
    plt.xlabel("X Meters")
    plt.plot(actual_x, actual_y, color="blue",label='actual', linestyle='-.', linewidth=1)
    plt.plot(x_values, y_values, color="green",label='predicted',linestyle='dotted', linewidth=2)
    plt.legend(bbox_to_anchor=(0.84,0.35))

    draw(to_draw)


# def animate(history: [model.Trilateration], ax=plt.axes()):
#    new = FuncAnimation(plt.gcf(), anim, len(history), fargs=(history, ax,), repeat=False)
# plt.show()
#    return new


# def anim(i, history: [model.Trilateration], ax):
#    x_values = []
#    y_values = []
#    for i in range(i + 1):
#        if i % 100 == 0:
#            item = history[i]
#            x_values.append(item.result.center.x)
#            y_values.append(item.result.center.y)
# for item in history[i].sniffers:
# create_point(item.center)
#    create_circle(history[i].result, target=True)
#    ax.plot(x_values, y_values, 'blue', linestyle='--')


def create_circle(circle: model.Circle, target=False):
    color = matplotlib.cm.jet(randint(50, 100))
    if target:
        color = matplotlib.cm.jet(1000)
    add_shape(plt.Circle((circle.center.x, circle.center.y), color=color, fill=False, zorder=1, radius=circle.radius,
                         alpha=0.8))
    plt.scatter(circle.center.x, circle.center.y, color=color, s=100, zorder=2)


def create_point(point: model.Point, color=matplotlib.cm.jet(randint(0, 00))):
    plt.scatter(float(point.x), float(point.y), color=color, s=100, zorder=2)


def add_shape(patch):
    ax = plt.gca()
    ax.add_patch(patch)
    plt.axis('scaled')


def draw(draw_list):
    for item in draw_list:
        if isinstance(item, model.Circle):
            create_circle(item)
        if isinstance(item, model.Point):
            create_point(item)
    plt.show()
