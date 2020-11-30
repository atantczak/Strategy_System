I've built a trading strategy analysis system on my local machine over the past few months and I've now transferred a generalized version of it (first draft) to GitHub. IE ~ I have a system that takes input data and trading signals and runs them through a simulation. However, I've built my system to test out, say, a MACD signal used for buying and selling. Obviously there are plenty of other trading signals you could use. Therefore, I've generalized the system to an extent and included an example signal such that those who are interested can adopt and adapt it to their liking.

Process for Using:
1) Run the Data_Save.py code such that the pricing data is saved to your local machine.
2) Consider what type of buying/selling signals you would like to test and ensure their accuracy and functionality within experimental code.
3) Consider if you want this experiment to run once, or several times, and change the iterative nature in the "run_exp()" function within the experimental code
to match your goals.
4) Ensure the output of the "run_exp()" function within the experimental code is in line with what you are hoping to learn. 
5) Run!

PS ~ Run-time optimization in the form of multiprocessing has been left out of this generalized version. Often times, when you're running an experiment/simulation that is going through hundreds of tickers, you'll want to ensure that your computer is using its full capacity. To do this takes careful consideration. This is a topic for a later project. 

The framework for this system is laid out below in this work flow image. 

![](Images/Workflow.png)
