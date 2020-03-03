from ortools.linear_solver import pywraplp

def main():
    # Create the linear solver with the GLOP backend.
    solver = pywraplp.Solver('simple_lp_program',
                             pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

    # Create the variables x and y.
    x = solver.NumVar(-500, 500, 'x')
    y = solver.NumVar(-500, 500, 'y')
    z = solver.NumVar(-500, 500, 'z')

    print('Number of variables =', solver.NumVariables())

    # Create a linear constraint, 0 <= x + y <= 2.
    #ct = solver.Constraint(0, 2, 'ct')
    #ct.SetCoefficient(x, 1)
    #ct.SetCoefficient(y, 1)
    
    ct = solver.Constraint(0, 1000, 'ct')
    ct.SetCoefficient(x, 1)
    ct.SetCoefficient(y, 1)

    print('Number of constraints =', solver.NumConstraints())

    # Create the objective function, 3 * x + y.
    objective = solver.Objective()
    objective.SetCoefficient(x, 1)
    objective.SetCoefficient(y, 1)
    objective.SetCoefficient(z, 1)
    objective.SetMaximization()

    solver.Solve()

    print('Solution:')
    print('Objective value =', objective.Value())
    print('x =', x.solution_value())
    print('y =', y.solution_value())
    print('z =', z.solution_value())


if __name__ == '__main__':
    main()