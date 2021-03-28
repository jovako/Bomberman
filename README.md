# bomberman_rl
Our Agents:

minimal1 to minimal8: rulebased trainers to generate data using our minimal features
nearest1 to nearest8: rulebased trainers to generate data using our subfield2 features
rewards1 to rewards8: rl agents saving the q values and exploring via rule_based actions


testnearest4: rl agent using the advanced rewards and subfield2 features
reduced: rl agent using the advanced rewards and minimal features, exploring via rule\_based

test4 to test11: agents trained on rule\_based data with number indicating range of vision, subfield1 features
testminimal: agents trained on minimal1 to minimal8 data
testrewards: agent trained on rewards1 to rewards8 data
testweights: agent trained on weighted rule\_based data, subfield2 features

testnearest5: agent trained on rule\_based data, vision 5, subfield2 features. Our final agent

rule: rule\_based\_agent, just renamed for easier calling

The feature and reward functions are imported from transform.py and rewards.py

The train....py files are used to train the networks on our data


