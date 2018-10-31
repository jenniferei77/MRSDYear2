import numpy as np
import pdb
import matplotlib.pyplot as plt

penalty = 0.5
T = 100;
num_hyps = 3
num_classes = 3

Die_Hard = 1
Pessimist = -1

def Divided(match):
    if match % 2:
        return -1
    else:
        return 1

def SleepImportance(sleep, no_tests):
    if sleep and no_tests:
        return 1
    elif not sleep and not no_tests:
        return -1
    elif (not sleep and no_tests) or (sleep and not no_tests):
        choose = np.random.randint(0,1)
        if choose:
            return 1
        else:
            return -1

def RecentWins(recent_wins, wins_ratio):
    if (recent_wins > 2) and (wins_ratio > 0.5):
        return 1
    elif (recent_wins < 2) and (wins_ratio < 0.5):
        return -1
    else:
        choose = np.random.randint(0,1)
        if choose:
            return 1
        else:
            return -1

def stochastic_extra(observes):
	total = 0
	if observes[0]:
		total += 50
	if observes[1]:
		total += 30
	if observes[2] > 2:
		total += 20
	if observes[3] > 0.5:
		total += 10
	if observes[4] % 2:
		total += 20

	if total >= 50:
		return 1
	else:
		return -1

def stochastic(t):
	num = np.random.randint(1,500)
	if (num*t % 2) == 0:
		return 1
	else: 
		return -1

def get_expert(class_type, weights, experts, alg_type, trial, observes):
    stochastic_choose = 0
    deterministic = 0
    adversarial = 0
    if class_type == 0:
    	return stochastic(trial)
        #return stochastic_extra(observes)
        pdb.set_trace()
    elif class_type == 1:
        deterministic = (sum(weights*experts) % 2)
        if deterministic:
            return deterministic
        else:
        	return -1
    elif class_type == 2:
    	if alg_type == 0:
    		if sum(experts*np.transpose(weights)) < 0:
    			return 1
    		else:
    			return -1
        else:
        	choose_i = np.random.multinomial(1, weights/(sum(weights)), size=1)
        	index_of_max = np.argmax(choose_i)
        	return -experts[index_of_max]
    
def weighted_majority(class_type, alg_type):
    weights = np.array([1] * num_hyps)
    avg_regret = []
    learner_loss = 0
    expert_loss = 0
    expert_losses = [0] * num_hyps
    wins_ratio = 0.
    recent_wins = 0.
    wins = 0.
    sleep = False
    no_tests = False
    expert0_loss = []
    expert1_loss = []
    expert2_loss = []
    expert3_loss = []
    expert4_loss = []
    for t in range(1,T):
    	sleep = np.random.randint(0,2)
    	no_tests = np.random.randint(0,2)
        #Receive x(t)
        experts = np.array([Die_Hard, Pessimist, Divided(t)])
        #experts = np.array([Die_Hard, Pessimist, Divided(t), RecentWins(recent_wins, wins_ratio), SleepImportance(sleep, no_tests)])

        #Get prediction, y_hat(t)
        if sum(experts*np.transpose(weights)) < 0: 
            prediction = -1 
        else: 
            prediction = 1

        observes = [sleep, no_tests, recent_wins, wins_ratio, t];
        #Receive y(t) 
        ground_truth = get_expert(class_type, weights, experts, alg_type, t, observes)
        
        if ground_truth == 1:
            wins += 1.0
            recent_wins += 1.0
        else:
            recent_wins = 0

        wins_ratio = wins/t

        #Update weights
        weights = weights * (1 - penalty * (ground_truth != experts))

        #Tally of each expert's mistakes
        expert_losses += ground_truth != experts
        expert0_loss.append(expert_losses[0])
        expert1_loss.append(expert_losses[1])
        expert2_loss.append(expert_losses[2])
        #expert3_loss.append(expert_losses[3])
        #expert4_loss.append(expert_losses[4])

        #Average regret
        learner_loss += (ground_truth != prediction)
        expert_loss = np.min(expert_losses)
        avg_regret.append((1./t)*(learner_loss - expert_loss))
    
    plt.title("Weighted Majority Algorithm Expert Losses")
    time = np.arange(1,T,1)
    plt.plot(time, expert0_loss, 'r')
    plt.plot(time, expert1_loss, 'b')
    plt.plot(time, expert2_loss, 'g')
    plt.legend(('Die Hard', 'Pessimist', 'Divided'))
    #plt.plot(time, expert3_loss, 'c')
    #plt.plot(time, expert4_loss, 'k')
    #plt.legend(('Die Hard', 'Pessimist', 'Divided', 'Winners', 'Sleeper'))
    plt.show()

    # if (class_type == 0):
	   # plt.figure(0)
	   # plt.title('Stochastic')
    # elif (class_type == 1):
	   # plt.figure(1)
	   # plt.title('Deterministic')
    # elif (class_type == 2):
	   # plt.figure(2)
	   # plt.title('Adversarial')			
    # time = np.arange(1,T,1)
    # plt.plot(time, avg_regret, 'b-')
    # plt.show()


def random_weighted_majority(class_type, alg_type):
    weights = np.array([1] * num_hyps)
    avg_regret = []
    learner_loss = 0
    expert_loss = 0
    expert_losses = [0] * num_hyps
    wins_ratio = 0.
    recent_wins = 0.
    wins = 0.
    expert0_loss = []
    expert1_loss = []
    expert2_loss = []
    expert3_loss = []
    expert4_loss = []

    for t in range(1,T):
    	sleep = np.random.randint(0,2)
    	no_tests = np.random.randint(0,2)
        #Receive x(t)
        experts = np.array([Die_Hard, Pessimist, Divided(t)])
        #experts = np.array([Die_Hard, Pessimist, Divided(t), RecentWins(recent_wins, wins_ratio), SleepImportance(sleep, no_tests)])

        #Get Multinomial 
        choose_i = np.random.multinomial(1, weights/(sum(weights)), size=1)

        #Get prediction, y_hat(t)
        index_of_max = np.argmax(choose_i)
        prediction = experts[index_of_max]

        observes = [sleep, no_tests, recent_wins, wins_ratio, t];
        #Receive y(t) 
        ground_truth = get_expert(class_type, weights, experts, alg_type, t, observes)

        #Update weights
        weights = weights * (1 - penalty * (ground_truth != experts))

        #Tally of each expert's mistakes
        expert_losses += ground_truth != experts
        #pdb.set_trace()
        expert0_loss.append(expert_losses[0])
        expert1_loss.append(expert_losses[1])
        expert2_loss.append(expert_losses[2])
        #expert3_loss.append(expert_losses[3])
        #expert4_loss.append(expert_losses[4])

        #Average regret
        learner_loss += (ground_truth != prediction)
        expert_loss = np.min(expert_losses)
        avg_regret.append((1./t)*(learner_loss - expert_loss))
    

    plt.title("Random Weighted Majority Algorithm Expert Losses")
    time = np.arange(1,T,1)
    #plt.plot(time, expert0_loss, 'r', time, expert1_loss, 'g', time, expert2_loss, 'b')
    plt.plot(time, expert0_loss, 'r')
    plt.plot(time, expert1_loss, 'b')
    plt.plot(time, expert2_loss, 'g')
    plt.legend(('Die Hard', 'Pessimist', 'Divided'))
    #plt.plot(time, expert3_loss, 'c')
    #plt.plot(time, expert4_loss, 'k')
    #plt.legend(('Die Hard', 'Pessimist', 'Divided', 'Winners', 'Sleeper'))
    plt.show()

    # if (class_type == 0):
    #     plt.figure(0)
    #     plt.title('Stochastic')
    # elif (class_type == 1):
    #     plt.figure(1)
    #     plt.title('Deterministic')
    # elif (class_type == 2):
    #     plt.figure(2)
    #     plt.title('Adversarial')            
    # time = np.arange(1,T,1)
    # plt.plot(time, avg_regret, 'b-')
    #plt.show()


def main():
    np.random.seed(222)
    for class_type in range(0, num_classes):
    	weighted_majority(class_type, 0)
        #random_weighted_majority(class_type, 1)
                          
                    
if __name__ == "__main__":
    main()
