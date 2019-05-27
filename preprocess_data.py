
import argparse
import config
import os
import json

if __name__ == '__main__':
  parse = argparse.ArgumentParser()
  parse.add_argument("corpus", choices=["en", "fr", "de", "ru", "pt", "zh", "pl", "uk", "ta"])
  args = parse.parse_args()

  inputdir = config.CORPUS_NAME_TO_PATH[args.corpus]
  outdir = os.path.join(inputdir, 'evidence')
  if not os.path.exists(outdir):
    os.mkdir(outdir)
  outdir2 = os.path.join(inputdir, 'qa')
  if not os.path.exists(outdir2):
    os.mkdir(outdir2)

  if os.path.basename(inputdir) == 'en':
    all_sess = ['train', 'dev', 'test']
  else:
    all_sess = ['dev', 'test']

  for sess in all_sess:
    f1 = os.path.join(inputdir, sess + '_doc.json')
    f2 = os.path.join(inputdir, sess + '.txt')
  
    fo = os.path.join(inputdir, 'qa', sess + '.json')
    outlines = []
    with open(f1, 'r') as fin1, open(f2, 'r') as fin2:
      for (qidx, (line1, line2)) in enumerate(zip(fin1, fin2)):
        if qidx % 100 == 0:
          print(qidx)
        qds = json.loads(line1.strip())
        qa = json.loads(line2.strip())
        out_item = {}
  
        out_item['answers'] = qa['answers']
        out_item['question_id'] = '%s_%d' % (sess, qidx)
        out_item['docs'] = []
        out_item['question'] = " ".join(qa['question'].split())

        for (didx, qd) in enumerate(qds):
          filename = "%s_%d_%d.txt" % (sess, qidx, didx)
          subdirpath = '%s_%d' % (sess, qidx)
          outd = os.path.join(outdir, subdirpath)
          if not os.path.isdir(outd):
            os.mkdir(outd)
          outfile = os.path.join(outd, filename)
  
          out_item['docs'].append((subdirpath, filename))
          document = "\n\n".join(qd['document'].split("\n\n"))
  
          with open(outfile, 'w') as fout:
            fout.write(document)
          
        outline = json.dumps(out_item, ensure_ascii=False)
        outlines.append(outline)
  
    with open(fo, 'w') as fout:
        for line in outlines:
            fout.write(line + "\n")

