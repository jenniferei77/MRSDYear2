import numpy as np
import pdb
import matplotlib.pyplot as plt

penalty = 0.5
T = 100;
num_hyps = 5
num_classes = 6

Die_Hard = 1
Pessimist = -1

def Divided(match):
    if match % 2:
        return -1
    else:
        return 1

def SleepImportance(sleep, no_tests):
    if sleep && no_tests:
        return 1
    elif !sleep && !no_tests:
        return -1
    elif (!sleep && no_tests) || (sleep && !no_tests):
        choose = np.random.randint(0,1)
        if choose:
            return 1
        else:
            return -1

def RecentWins(recent_wins, wins_ratio)
    if (recent_wins > 2) && (wins_ratio > 0.5):
        return 1
    elif (recent_wins < 2) && (wins_ratio < 0.5):
        return -1
    else:
        choose = np.random.randint(0,1)
        if choose:
            return 1
        else:
            return -1

#Nature Sends Observation
def get_expert(class_type, weights, experts, prediction, recent_wins, wins_ratio):
    stochastic_choose = 0
    deterministic = 0
    adversarial = 0
    if class_type == 0:
        stochastic_choose = np.random.randint(0,1)
        if stochastic_choose:
            return stochastic_choose
        else:
            return -1
    elif class_type == 1:
        deterministic = (sum(weights*experts) % 2)
        if deterministic:
            return deterministic
	    else:
	        return -1
    elif class_type == 2:
        adversarial = -prediction
        return adversarial

    
def weighted_majority(class_type):
    weights = np.array([1] * num_hyps)
    avg_regret = []
    learner_loss = 0
    expert_loss = 0
    expert_losses = [0] * num_hyps
    wins_ratio = 0
    recent_wins = 0
    for t in range(1,T):
        #Receive x(t)
        experts = np.array([Die_Hard, Pessimist, Divided(t), RecentWins(recent_wins, wins_ratio)])

        #Get prediction, y_hat(t)
        if sum(experts*np.transpose(weights)) < 0: 
            prediction = -1 
        else: 
            prediction = 1

        #Receive y(t) 
        ground_truth = get_expert(class_type, weights, experts, prediction, recent_wins, wins_ratio)
        if ground_truth:
            wins++
            recent_wins++
        else:
            recent_wins = 0

        wins_ratio = wins/t.


        #Update weights
        weights = weights * (1 - penalty * (ground_truth != experts))

        #Tally of each expert's mistakes
        expert_losses += ground_truth != experts

        #Average regret
        learner_loss += (ground_truth != prediction)
        expert_loss = np.min(expert_losses)
        avg_regret.append((1./t)*(learner_loss - expert_loss))

        #pdb.set_trace()
    
    if (class_type == 0):
	   plt.figure(0)
	   plt.title('Stochastic')
    elif (class_type == 1):
	   plt.figure(1)
	   plt.title('Deterministic')
    elif (class_type == 2):
	   plt.figure(2)
	   plt.title('Adversarial')			
    time = np.arange(1,T,1)
    plt.plot(time, avg_regret, 'bo')
    plt.show()


def random_weighted_majority(class_type):
    weights = np.array([1] * num_hyps)
    avg_regret = []
    learner_loss = 0
    expert_loss = 0
    expert_losses = [0] * num_hyps
    for t in range(1,T):
        #Receive x(t)
        experts = np.array([Die_Hard, Pessimist, Divided(t)])

        #Get Multinomial 
        choose_i = np.random.multinomial(1, weights/(sum(weights)), size=1)

        #Get prediction, y_hat(t)
        index_of_max = np.argmax(choose_i)
        prediction = experts[index_of_max]

        #Receive y(t) 
        ground_truth = get_expert(class_type, weights, experts, prediction)

        #Update weights
        weights = weights * (1 - penalty * (ground_truth != experts))

        #Tally of each expert's mistakes
        expert_losses += ground_truth != experts

        #Average regret
        learner_loss += (ground_truth != prediction)
        expert_loss = np.min(expert_losses)
        avg_regret.append((1./t)*(learner_loss - expert_loss))

        #pdb.set_trace()
    
    if (class_type == 0):
        plt.figure(0)
        plt.title('Stochastic')
    elif (class_type == 1):
        plt.figure(1)
        plt.title('Deterministic')
    elif (class_type == 2):
        plt.figure(2)
        plt.title('Adversarial')            
    time = np.arange(1,T,1)
    plt.plot(time, avg_regret, 'bo')
    plt.show()



def main():
    np.random.seed(222)
    for class_type in range(0, num_classes):
        random_weighted_majority(class_type)
                          
                        
    
if __name__ == "__main__":
    main()
