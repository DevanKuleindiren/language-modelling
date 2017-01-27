#import "RunModelViewController.h"

#include <list>
#include <string>
#include "tensorflow/Source/lm/rnn/rnn.h"

void LoadAndTestRNN();

@interface RunModelViewController ()
@end

@implementation RunModelViewController {
}

- (IBAction)loadRNN:(id)sender {
    LoadAndTestRNN();
}

@end

NSString* FilePathForResourceName(NSString* name, NSString* extension) {
    NSString* file_path = [[NSBundle mainBundle] pathForResource:name ofType:extension];
    if (file_path == NULL) {
        LOG(FATAL) << "Couldn't find '" << [name UTF8String] << "." << [extension UTF8String] << "' in bundle.";
    }
    return file_path;
}

void LoadAndTestRNN() {
    NSString* model_path = FilePathForResourceName(@"graph", @"pb");
    
    // Strip the end of the file path to make it a directory path.
    NSRange range = [model_path rangeOfString: @"/" options: NSBackwardsSearch];
    NSString* model_dir = [model_path substringToIndex: range.location];
    
    std::list<std::string> seq = std::list<std::string>({"it", "is"});
    RNN *rnn = new RNN(std::string([model_dir UTF8String]));
    LOG(INFO) << rnn->Prob(seq);
}
