
# coding: utf-8

# In[1]:


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns


# -Each location can only hold 20 cars.
# -Every time a car is rented, we earn $10 (Reward)
# -Every time we move a car overnight to another location, it costs us $2 (Negative Reward).
# -The maximum number of cars we can move overnight is 5 (Action).
# -The number of cars requested and returned at each location (n) on any given day are Poisson random variables.
# -The expected number (lambda) of rental requests at the first and second location is 3 and 4 respectively.
# -The expected number of rental returns at the first and second location is 3 and 2 respectively.
# --Thus, the second location has more rentals than returns, whereas the first location has an equal number of rentals as returns.
# -Our discount rate for future returns, (γ), is 0.9.
# -The time step are days (thus, one step in an iteration can be considered a full day), the state is the number of cars at each location at the end of the day, and the actions are the net number of cars moved between the two locations overnight.

# In[2]:


MAX_CARS = 20 # maximum # of cars in each location
MAX_MOVE_OF_CARS = 5 # maximum # of cars to move during night
RENTAL_CREDIT = 10 # credit earned by a car
MOVE_CAR_COST = 2 # cost of moving a car

RENTAL_REQUEST_FIRST_LOC = 3 #expectation for rental requests in first location
RENTAL_REQUEST_SECOND_LOC = 4 # expectation for rental requests in second location
RETURNS_FIRST_LOC = 3 # expectation for # of cars returned in first location
RETURNS_SECOND_LOC = 2 # expectation for # of cars returned in second location

DISCOUNT = 0.9


# Number between -5 and 5, where positive numbers indicate moving cars from Location 1 to Location 2, and negative numbers indicate moving cars from Location 2 to Location 1

# In[3]:


# all possible actions
actions = np.arange(-MAX_MOVE_OF_CARS, MAX_MOVE_OF_CARS + 1)


# The Poisson Distribution focuses on the number of discrete events or occurrences over a specified interval or continuum (time, length, distance, etc.). A poisson random variable, x, is the number of events in a given unit of time, which can be any non-negative whole value

# In[4]:


# An up bound for poisson distribution
# If n is greater than this value, then the probability of getting n is truncated to 0
POISSON_UPPER_BOUND = 11

# Probability for poisson distribution
# @lam: lambda should be less than 10 for this function
poisson_cache = dict()


def poisson_probability(n, lam):
    global poisson_cache
    key = n * 10 + lam
    if key not in poisson_cache:
        poisson_cache[key]=np.exp(-lam) * pow(lam,n) / math.factorial(n)
    return poisson_cache[key]


# state: # of cars in first location, # of cars in second location
# action: positive if moving cars from first location to second location,
#             negative if moving cars from second location to first location
# stateValue: state value matrix
# constant_returned_cars:  if set True, model is simplified such that the # of cars returned in daytime becomes constant rather than a random value from poisson distribution, which will reduce calculation time and leave the optimal policy/value state matrix almost the same

# In[5]:


def expected_return(state, action, state_value, constant_returned_cars):

    # initailize total return
    returns = 0.0

    # cost for moving cars
    returns -= MOVE_CAR_COST * abs(action)

    # moving cars
    NUM_OF_CARS_FIRST_LOC = min(state[0] - action, MAX_CARS)
    NUM_OF_CARS_SECOND_LOC = min(state[1] + action, MAX_CARS)

    # go through all possible rental requests
    for rental_request_first_loc in range(POISSON_UPPER_BOUND):
        for rental_request_second_loc in range(POISSON_UPPER_BOUND):
            # probability for current combination of rental requests
            prob = poisson_probability(rental_request_first_loc, RENTAL_REQUEST_FIRST_LOC) *                 poisson_probability(rental_request_second_loc, RENTAL_REQUEST_SECOND_LOC)

            num_of_cars_first_loc = NUM_OF_CARS_FIRST_LOC
            num_of_cars_second_loc = NUM_OF_CARS_SECOND_LOC

            # valid rental requests should be less than actual # of cars
            valid_rental_first_loc = min(num_of_cars_first_loc, rental_request_first_loc)
            valid_rental_second_loc = min(num_of_cars_second_loc, rental_request_second_loc)

            # get credits for renting
            reward = (valid_rental_first_loc + valid_rental_second_loc) * RENTAL_CREDIT
            num_of_cars_first_loc -= valid_rental_first_loc
            num_of_cars_second_loc -= valid_rental_second_loc

            if constant_returned_cars:
                # get returned cars, those cars can be used for renting tomorrow
                returned_cars_first_loc = RETURNS_FIRST_LOC
                returned_cars_second_loc = RETURNS_SECOND_LOC
                num_of_cars_first_loc = min(num_of_cars_first_loc + returned_cars_first_loc, MAX_CARS)
                num_of_cars_second_loc = min(num_of_cars_second_loc + returned_cars_second_loc, MAX_CARS)
                returns += prob * (reward + DISCOUNT * state_value[num_of_cars_first_loc, num_of_cars_second_loc])
            else:
                for returned_cars_first_loc in range(POISSON_UPPER_BOUND):
                    for returned_cars_second_loc in range(POISSON_UPPER_BOUND):
                        prob_return = poisson_probability(
                            returned_cars_first_loc, RETURNS_FIRST_LOC) * poisson_probability(returned_cars_second_loc, RETURNS_SECOND_LOC)
                        num_of_cars_first_loc_ = min(num_of_cars_first_loc + returned_cars_first_loc, MAX_CARS)
                        num_of_cars_second_loc_ = min(num_of_cars_second_loc + returned_cars_second_loc, MAX_CARS)
                        prob_ = prob_return * prob
                        returns += prob_ * (reward + DISCOUNT *
                                            state_value[num_of_cars_first_loc_, num_of_cars_second_loc_])
    return returns



# In[6]:


def figureJC(constant_returned_cars=True):
    value = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
    policy = np.zeros(value.shape, dtype=np.int)

    iterations = 0
    _, axes = plt.subplots(2, 3, figsize=(40, 20))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()
    while True:
        fig = sns.heatmap(np.flipud(policy), cmap="BuPu", ax=axes[iterations])
        fig.set_ylabel('# cars at first location', fontsize=30)
        fig.set_yticks(list(reversed(range(MAX_CARS + 1))))
        fig.set_xlabel('# cars at second location', fontsize=30)
        fig.set_title('policy {}'.format(iterations), fontsize=30)

        # policy evaluation (in-place)
        while True:
            old_value = value.copy()
            for i in range(MAX_CARS + 1):
                for j in range(MAX_CARS + 1):
                    new_state_value = expected_return([i, j], policy[i, j], value, constant_returned_cars)
                    value[i, j] = new_state_value
            max_value_change = abs(old_value - value).max()
            print('max value change {}'.format(max_value_change))
            if max_value_change < 1e-4:
                break

        # policy improvement
        policy_stable = True
        for i in range(MAX_CARS + 1):
            for j in range(MAX_CARS + 1):
                old_action = policy[i, j]
                action_returns = []
                for action in actions:
                    if (0 <= action <= i) or (-j <= action <= 0):
                        action_returns.append(expected_return([i, j], action, value, constant_returned_cars))
                    else:
                        action_returns.append(-np.inf)
                new_action = actions[np.argmax(action_returns)]
                policy[i, j] = new_action
                if policy_stable and old_action != new_action:
                    policy_stable = False
        print('policy stable {}'.format(policy_stable))

        if policy_stable:
            fig = sns.heatmap(np.flipud(value), cmap="BuPu", ax=axes[-1])
            fig.set_ylabel('# cars at first location', fontsize=30)
            fig.set_yticks(list(reversed(range(MAX_CARS + 1))))
            fig.set_xlabel('# cars at second location', fontsize=30)
            fig.set_title('optimal value', fontsize=30)
            break

        iterations += 1

    plt.savefig('figureJacksCar.png')
    plt.close()


# In[7]:


if __name__ == '__main__':
    figureJC()

