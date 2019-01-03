import _pickle as pickle 
import os
import sys
sys.path.insert(1,'../coco-caption/')
from pycocoevalcap.bleu.bleu import Bleu 
from pycocoevalcap.rouge.rouge import Rouge 
from pycocoevalcap.cider.cider import Cider  
from pycocoevalcap.meteor.meteor import Meteor 

def score(ref, hypo):
	scorers = [
		(Bleu(4),["Bleu_1","Bleu_2", "Bleu_3", "Bleu_4"]),
		(Meteor(), "METEOR"),
		(Rouge(), "ROUGE"),
		(Cider(),"CIDER")]

	final_scores = {}
	for scorer, method in scorers:
		score, scores = scorer.compute_score(ref, hypo)
		if type(score)==list:
			for m,s in zip(method, score):
				final_scores[m] = s
		else:
			final_scores[method] = score
	return final_scores

def evaluate(data_path='./data', split='val', get_scores=False):
	reference_path = os.path.join(data_path, "%s%s.references.pkl"%(split,split))
	candidate_path = os.path.join(data_path, "%s%s.candidate.captions.pkl"%(split,split))

	with open(reference_path, 'rb') as f:
		ref = pickle.load(f)
	with open(candidate_path, 'rb') as f:
		cand = pickle.load(f) 


	hypo = {}
	for i, caption in enumerate(cand):
		hypo[i] = [caption]

	score_final = score(ref, hypo)

	print('Bleu_1:\t',score_final['Bleu_1'])
	print('Bleu_2:\t',score_final['Bleu_2'])
	print('Bleu_3:\t',score_final['Bleu_3'])
	print('Bleu_4:\t',score_final['Bleu_4'])
	print('METEOR:\t',score_final['METEOR'])
	print('ROUGE:\t',score_final['ROUGE'])
	print('CIDER:\t',score_final['CIDER'])

	if get_scores:
		return score_final

