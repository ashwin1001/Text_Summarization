from rouge import Rouge


def rouge_score(summary,gold_sumary):
    hypothesis = summary
    reference = gold_sumary
    rouge = Rouge()
    scores = rouge.get_scores(reference, hypothesis)
    print scores
    print 'Rouge-2 scores:\t',
    print ' Precision: ',
    print scores[0]['rouge-2']['p'],
    print ' Recall: ',
    print scores[0]['rouge-2']['r'],
    print ' F_measure: ',
    print scores[0]['rouge-2']['f'],


if __name__=="__main__":
    f=open('summary/summary50.3TR.txt','r')
    summary=f.read()
    f1=open('GroundTruth/Topic5.1','r')
    gold_summ=f1.read()
    rouge_score(summary,gold_summ)

