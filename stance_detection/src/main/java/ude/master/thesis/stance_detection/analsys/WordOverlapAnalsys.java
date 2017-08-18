package ude.master.thesis.stance_detection.analsys;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import ude.master.thesis.stance_detection.util.StanceDetectionDataReader;

public class WordOverlapAnalsys {
	
	public static void main(String[] args) throws IOException {
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, false);
		Map<Integer, String> bodyIdMap = sddr.getTrainIdBodyMap();
		List<List<String>> stances = sddr.getTrainStances();
		
		
		
	}

}
