package ude.master.thesis.stance_detection.util;

import java.io.FileReader;
import java.io.IOException;
import java.io.LineNumberReader;
import java.util.ArrayList;

import org.clapper.util.misc.FileHashMap;

public class TestWithPPDB {

	public static void main(String[] args) throws IOException {
		
		extractParaphrases("C:/Master UDE/thesis/software/ppdb-2.0-xxl-all/ppdb-2.0-xxl-all", "C:/Master UDE/thesis/software/all_ppdb_xxl");
		
	}
	
	public static void extractParaphrases(String ppdbOriginalFilename, String extractedPPDBFilename)
			throws IOException {
		FileReader fr = null;
		LineNumberReader lnr = null;
		String str;
		int i;

		try {

			// create new reader
			fr = new FileReader(ppdbOriginalFilename);
			lnr = new LineNumberReader(fr);

			FileHashMap<String, ArrayList<ArrayList<String>>> ppdbScore = new FileHashMap<String, ArrayList<ArrayList<String>>>(
					extractedPPDBFilename, FileHashMap.FORCE_OVERWRITE);

			// read lines till the end of the stream
			int j = 0;
			// Set<String> entailments = new HashSet<>();
			while ((str = lnr.readLine()) != null) {
				i = lnr.getLineNumber();
				// System.out.print("("+i+")");

				// prints string
				System.out.println(str);

				String[] data = str.split(" \\|\\|\\| ");
				String textLhs = data[1];
				String textRhs = data[2];

				// System.out.println(data[3]);
				// System.out.println(data[3].split(" ")[1]);
				// System.out.println(data[3].split(" ")[0]);
				double ppdb2Score = Double.valueOf(data[3].split(" ")[0].split("=")[1]);
				String entailment = data[data.length - 1].trim();

				// entailments.add(entailment);

				String key = textLhs;
				ArrayList<String> value = new ArrayList<>();
				value.add(textRhs);
				value.add(Double.toString(ppdb2Score));
				value.add(entailment);

				/*if (textLhs.equals("neighbouring")) {
					System.out.println(textRhs + " " + ppdb2Score + " " + entailment);
					// System.out.println(str);
				}*/

				// if (entailment.contains("ReverseEntailment"))
				// System.out.println(textLhs + " " + textRhs);
				// System.out.println(textLhs + " " + textRhs + " " + ppdb2Score
				// + " " + entailment);

				j++;

				// if(j==3)
				// break;
				if (ppdbScore.containsKey(key)) {
					ArrayList<ArrayList<String>> values = ppdbScore.get(key);
					values.add(value);
					ppdbScore.put(key, values);

				} else {
					ArrayList<ArrayList<String>> paphrases = new ArrayList<>();
					paphrases.add(value);
					ppdbScore.put(key, paphrases);
				}

			}
			//System.out.println(ppdbScore.get("neighbouring"));
			//System.out.println(j);
			// System.out.println(entailments);
			ppdbScore.save();
			ppdbScore.close();
		} catch (Exception e) {

			// if any error occurs
			e.printStackTrace();
		} finally {

			// closes the stream and releases system resources
			if (fr != null)
				fr.close();
			if (lnr != null)
				lnr.close();
		}
	}
}
