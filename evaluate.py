import matplotlib.pyplot as plt
import taxiagent
import randomagent
import csv


# Calculate the scores of both agents
random_agent_scores = randomagent.randomagent(verbose=True)
q_agent_scores = taxiagent.taxiagent(verbose=True)


# Save the scores of both agents as csv
with open("random_scores.csv", "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for v in random_agent_scores:
        writer.writerow([v])

with open("q_scores.csv", "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for v in q_agent_scores:
        writer.writerow([v])


# Plot all agents
plt.plot(random_agent_scores, "ko", markersize=0.3)
plt.title("Random Agent")
plt.xlabel('Episodes')
plt.ylabel('Score')
plt.show()

plt.plot(q_agent_scores, "ko", markersize=0.3)
plt.title("Q-learning Agent")
plt.xlabel('Episodes')
plt.ylabel('Score')
plt.show()
