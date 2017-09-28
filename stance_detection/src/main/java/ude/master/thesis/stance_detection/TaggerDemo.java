package ude.master.thesis.stance_detection;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.StringReader;
import java.util.List;

import edu.stanford.nlp.ling.SentenceUtils;
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.process.Morphology;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;

public class TaggerDemo  {

 

  private TaggerDemo() {}

  public static void main(String[] args) throws Exception {

    MaxentTagger tagger = new MaxentTagger("resources/taggers/english-left3words-distsim.tagger");
    List<List<HasWord>> sentences = MaxentTagger.tokenizeText(new StringReader("The sun is shining."));
    for (List<HasWord> sentence : sentences) {
      List<TaggedWord> tSentence = tagger.tagSentence(sentence);
      System.out.println(SentenceUtils.listToString(tSentence, false));
    }
  }

}
