import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

public class Check {

    // This code takes in two files and checks if they are exactly similar

    public static void main(String[] args) throws IOException {

        if (args.length != 1) {
            System.out.println("Running pre-defined test case:");
            File file1 = new File("small_model_correct.txt");
            File file2 = new File("small_model.txt");

            BufferedReader br1 = new BufferedReader(new FileReader(file1));
            BufferedReader br2 = new BufferedReader(new FileReader(file2));

            // Compare these two
            String st1, st2;
            while ((st1 = br1.readLine()) != null && (st2 = br2.readLine()) != null) {
                if (!st1.equals(st2)) {
                    System.out.println("Files are not equal");
                    return;
                }
            }
            System.out.println("Files are equal");
            System.exit(0);
            System.exit(0);
        }

        File file1 = new File(args[0]);
        File file2 = new File(args[1]);
        BufferedReader br1 = new BufferedReader(new FileReader(file1));
        BufferedReader br2 = new BufferedReader(new FileReader(file2));
        String st1, st2;
        while ((st1 = br1.readLine()) != null && (st2 = br2.readLine()) != null) {
            if (!st1.equals(st2)) {
                System.out.println("Files are not equal");
                return;
            }
        }
        if (br1.readLine() != null || br2.readLine() != null) {
            System.out.println("Files are not equal");
            return;
        }
        System.out.println("Files are equal");
    }

}