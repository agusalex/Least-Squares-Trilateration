
# easy-trilateration
Trilateration example using least squares method in scipy (Graphing tools included).

![](https://github.com/agusalex/easy-trilateration/blob/master/images/img2.png?raw=true)

Trilateration enables the unknown point to be found. However a since there are a number of samples a non linear least squares method needs to be used to find the solution that has the least error. 

It is distinct from triangulation which has a series of angles to an unknown point. Trilateration uses a series of distances to an unkown point.

## How to install

    pip install easy_trilateration


## How to use
### Simple Trilateration
    from easy_trilateration.model import *  
    from easy_trilateration.least_squares import easy_least_squares  
    from easy_trilateration.graph import *  
      
    if __name__ == '__main__':  
        arr = [Circle(100, 100, 50),  
      Circle(100, 50, 50),  
      Circle(50, 50, 50),  
      Circle(50, 100, 50)]  
        result, meta = easy_least_squares(arr)  
        create_circle(result, target=True)  
        draw(arr)
### Location History
    arr = Trilateration([Circle(100, 100, 70.5),
                         Circle(100, 50, 50),
                         Circle(50, 50, 0),
                         Circle(50, 100, 50)])

    arr2 = Trilateration([Circle(100, 100, 50),
                          Circle(100, 50, 70.5),
                          Circle(50, 50, 50),
                          Circle(50, 100, 0)])

    arr3 = Trilateration([Circle(100, 100, 0),
                          Circle(100, 50, 50),
                          Circle(50, 50, 70.5),
                          Circle(50, 100, 50)])

    arr4 = Trilateration([Circle(100, 100, 50),
                          Circle(100, 50, 0),
                          Circle(50, 50, 50),
                          Circle(50, 100, 70.5)])

    hist: [Trilateration] = [arr, arr2, arr3, arr4, arr]

    solve_history(hist)

    a = animate(hist)

![](https://github.com/agusalex/easy-trilateration/blob/master/images/img3.png?raw=true)

## Method
This code uses the [scipy.optimize.least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html) method.
