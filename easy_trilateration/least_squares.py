import random

from easy_trilateration.model import *
from scipy.optimize import least_squares


def solve_history(history: [Trilateration]):
    guess = Circle(0, 0, 0)
    for item in history:
        guess = solve(item, guess)


def solve(trilateration: Trilateration, guess: Circle = Circle(0, 0, 0)) -> Circle:
    result, meta = easy_least_squares(trilateration.sniffers, guess)
    trilateration.result = result
    return result

def easy_least_squares(crls, guess=Circle(0, 0, 0)):
    g = (guess.center.x, guess.center.y, guess.radius)
    result = least_squares(equations, g, args=[crls])
    xf, yf, rf = result.x

    return Circle(xf, yf, rf), result


def equations(guess, crls: [Circle]):
    eqs = []
    x, y, r = guess
    for circle in crls:
        eqs.append(((x - circle.center.x) ** 2 + (y - circle.center.y) ** 2 - (circle.radius - r) ** 2))
    return eqs
