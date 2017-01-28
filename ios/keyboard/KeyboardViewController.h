//
//  KeyboardViewController.h
//  keyboard
//
//  Created by Devan Kuleindiren on 27/01/2017.
//  Copyright © 2017 Google. All rights reserved.
//

#import <UIKit/UIKit.h>
#include "rnn.hpp"

@interface KeyboardViewController : UIInputViewController {
    IBOutlet UILabel *firstPrediction;
    IBOutlet UILabel *secondPrediction;
    IBOutlet UILabel *thirdPrediction;
    bool caps_on;
    
    RNN2 *rnn;
}

- (void)insertString:(NSString *)s;

- (IBAction)keyPress:(id)sender;
- (IBAction)newLine:(id)sender;
- (IBAction)caps:(id)sender;
- (IBAction)backspace:(id)sender;
- (IBAction)nextKeyboard:(id)sender;

@end
