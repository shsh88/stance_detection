package ude.master.thesis.stance_detection.analsys;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.opencsv.CSVWriter;

import ude.master.thesis.stance_detection.processor.FeatureExtractor;
import ude.master.thesis.stance_detection.processor.Lemmatizer;
import ude.master.thesis.stance_detection.util.StanceDetectionDataReader;

public class WordOverlapAnalsys {

	public static void main(String[] args) throws IOException {
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, false);
		Map<Integer, String> bodyIdMap = sddr.getTrainIdBodyMap();
		List<List<String>> stances = sddr.getTrainStances();
		
		//System.out.println(bodyIdMap == null);
		//System.out.println(stances == null);

		List<Map<String, Integer>> wordsPos = new ArrayList<>();

		List<String[]> entries = new ArrayList<>();

		int i = 0;
		/*for (List<String> stance : stances) {
			//System.out.println("1111");
			List<String> entry = new ArrayList<>();
			entry.addAll(stance);

			String h = stance.get(0);
			String b = bodyIdMap.get(Integer.valueOf(stance.get(1)));
			//System.out.println(h+b);
			Set<String> words = FeatureExtractor.getOverlappedWords(h, b);
			//System.out.println(words);
			Map<String, Integer> wordsMap = new HashMap<>();
			String wordsConcat = "";
			for (String w : words) {
				//System.out.println("222");
				List<String> bodyLemm = FeatureExtractor.removeStopWords(new Lemmatizer().lemmatize(FeatureExtractor.clean(b)));
				
				if (bodyLemm.contains(w)) {
					//System.out.println("****");
					wordsMap.put(w, bodyLemm.indexOf(w));
					wordsConcat += w + " " + bodyLemm.indexOf(w)+ ",";
				}

			}
			wordsConcat = wordsConcat.substring(0, wordsConcat.lastIndexOf(','));
			//System.out.println(wordsConcat);
			wordsPos.add(wordsMap);
			entry.add("" + words.size());
			entry.add(wordsConcat);
			entries.add(entry.toArray(new String[entry.size()]));
			
			//System.out.println("i= " + i);
			
			i++;
			if(i%1000 == 0)
				System.out.println("processed : " + i);
		}
		
		String fileName = "C:/arff_data/analysis_wordoverlap.csv";

        try (CSVWriter writer = new CSVWriter(new FileWriter(fileName))) {
            writer.writeAll(entries);
        }*/
		
		
		for (List<String> stance : stances) {
			
		}

	}

}
