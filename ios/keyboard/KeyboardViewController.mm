//
//  KeyboardViewController.m
//  keyboard
//
//  Created by Devan Kuleindiren on 27/01/2017.
//  Copyright © 2017 Google. All rights reserved.
//

#import "KeyboardViewController.h"

#include <list>
#include <string>


@interface KeyboardViewController ()
@end

@implementation KeyboardViewController

- (void)updateViewConstraints {
    [super updateViewConstraints];
    
    // Add custom view sizing constraints here
    if (self.view.frame.size.width == 0 || self.view.frame.size.height == 0)
        return;

    CGFloat _expandedHeight = 252;

    NSLayoutConstraint *_heightConstraint = [NSLayoutConstraint constraintWithItem:self.view attribute:NSLayoutAttributeHeight relatedBy:NSLayoutRelationEqual toItem:nil attribute:NSLayoutAttributeNotAnAttribute multiplier:0.0 constant: _expandedHeight];

    [self.view addConstraint: _heightConstraint];
}

- (id) init {
    self = [super init];
    if (self) {
        // Init here.
    }
    
    return self;
}

- (void)viewDidLoad {
    [super viewDidLoad];
    
    usePrevStateRNN = false;

    predictionButtons = [[NSArray alloc] initWithObjects:firstPrediction, secondPrediction, thirdPrediction, nil];
    
    // Initialise the shift button.
    shiftButtonState = SHIFT;
    UITapGestureRecognizer *shiftSingleTap = [[UITapGestureRecognizer alloc] initWithTarget:self action:@selector(shiftSingleTap)];
    shiftSingleTap.numberOfTapsRequired = 1;
    [shiftButton addGestureRecognizer:shiftSingleTap];

    UITapGestureRecognizer *shiftDoubleTap = [[UITapGestureRecognizer alloc] initWithTarget:   self action:@selector(shiftDoubleTap)];
    shiftDoubleTap.numberOfTapsRequired = 2;
    [shiftButton addGestureRecognizer:shiftDoubleTap];

    // Key pad state is initially showing the letters.
    padState = LETTERS;
    symbolPairs1 = @{
                     @"1" : @"[",
                     @"2" : @"]",
                     @"3" : @"{",
                     @"4" : @"}",
                     @"5" : @"#",
                     @"6" : @"%",
                     @"7" : @"^",
                     @"8" : @"*",
                     @"9" : @"+",
                     @"0" : @"=",
                     @"-" : @"_",
                     @"/" : @"\\",
                     @":" : @"|",
                     @";" : @"~",
                     @"(" : @"<",
                     @")" : @">",
                     @"£" : @"€",
                     @"&" : @"$",
                     @"@" : @"¥",
                     @"\"" : @"•",
                     @"." : @".",
                     @"," : @",",
                     @"?" : @"?",
                     @"!" : @"!",
                     @"'" : @"'",
                     };
    symbolPairs2 = @{
                     @"[" : @"1",
                     @"]" : @"2",
                     @"{" : @"3",
                     @"}" : @"4",
                     @"#" : @"5",
                     @"%" : @"6",
                     @"^" : @"7",
                     @"*" : @"8",
                     @"+" : @"9",
                     @"=" : @"0",
                     @"_" : @"-",
                     @"\\" : @"/",
                     @"|" : @":",
                     @"~" : @";",
                     @"<" : @"(",
                     @">" : @")",
                     @"€" : @"£",
                     @"$" : @"&",
                     @"¥" : @"@",
                     @"•" : @"\"",
                     @"." : @".",
                     @"," : @",",
                     @"?" : @"?",
                     @"!" : @"!",
                     @"'" : @"'",
                     };

    // Load RNN.
    NSString* model_path = [[NSBundle mainBundle] pathForResource:@"graph" ofType:@"pb"];
    NSRange range = [model_path rangeOfString: @"/" options: NSBackwardsSearch];
    rnn = new RNN(std::string([[model_path substringToIndex: range.location] UTF8String]));
    
    charTrie = new CharTrie();
    std::list<std::string> blackList = std::list<std::string>({".", ",", "?", "!", "'", "\"", "/", "\\", ":", ";", "(", ")", "{", "}", "@", "#", "£", "$", "%", "^", "&", "*", "-", "_", "+", "=", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"});
    for (std::unordered_map<std::string, size_t>::const_iterator it = rnn->GetVocab()->begin(); it != rnn->GetVocab()->end(); ++it) {
        if (!(it->first.compare("<unk>") == 0 ||
              it->first.compare("N") == 0 ||
              it->first.compare("<s>") == 0)) {
            bool containsBlacklistedChar = false;
            for (std::list<std::string>::iterator itBl = blackList.begin(); itBl != blackList.end(); ++itBl) {
                if (it->first.find(*itBl) != std::string::npos) {
                    containsBlacklistedChar = true;
                    break;
                }
            }
            if (!containsBlacklistedChar) {
                charTrie->Insert(it->first, 0);
            }
        }
    }
    rnn->ProbAllFollowing(std::list<std::string>({"<s>"}), charTrie, false);
    std::list<std::pair<std::string, double>> top3 = charTrie->GetMaxKWithPrefix("", 3);
    [self setPredictionsWithTop3:&top3];

    // Round edges on all buttons.
    for(UIView *v in [self.view subviews]) {
        if ([v isKindOfClass:[UIButton class]]) {
            [[(UIButton *)v layer] setCornerRadius:5];
            [(UIButton *)v setClipsToBounds:true];
        } else if ([v isKindOfClass:[UIView class]]) {
            for(UIView *v_sub in [(UIView *)v subviews]) {
                if ([v_sub isKindOfClass:[UIButton class]]) {
                    [[(UIButton *)v_sub layer] setCornerRadius:5];
                    [(UIButton *)v_sub setClipsToBounds:true];
                }
            }
        }
    }
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated
}

- (void)textWillChange:(id<UITextInput>)textInput {
    // The app is about to change the document's contents. Perform any preparation here.
}

- (void)textDidChange:(id<UITextInput>)textInput {
    // The app has just changed the document's contents, the document context has been updated.
}

- (void)insertString:(NSString *)s {
    [self.textDocumentProxy insertText:s];
}

-(IBAction)keyPress:(id)sender {
    if (![sender isKindOfClass:[UIButton class]]) {
        return;
    }
    [self insertString:[(UIButton *)sender currentTitle]];
    NSArray *tokens = [self.textDocumentProxy.documentContextBeforeInput componentsSeparatedByString:@" "];
    unsigned long numTokens = [tokens count];
    if (numTokens > 1 && [[tokens objectAtIndex:(numTokens - 2)] length] > 0 && [[tokens lastObject] length] == 0) {
        [self newPredictions];
    } else {
        [self updatePredictions];
    }
    if (shiftButtonState == SHIFT) {
        [self switchAllKeys:false];
        shiftButtonState = LOWER;
        [shiftButton setBackgroundImage:[UIImage imageNamed:@"Lower.png"] forState:UIControlStateNormal];
    }
}

- (IBAction)newLine:(id)sender {
    [self insertString:@"\n"];
}

- (IBAction)predictWord:(id)sender {
    if (![sender isKindOfClass:[UIButton class]]) {
        return;
    }
    
    NSArray *tokens = [self.textDocumentProxy.documentContextBeforeInput componentsSeparatedByString:@" "];
    for (int i = 0; i < [[tokens lastObject] length];i++) {
        [self.textDocumentProxy deleteBackward];
    }
    [self insertString:[NSString stringWithFormat:@"%@ ", [(UIButton *)sender currentTitle]]];
    [self newPredictions];
}

- (void)shiftSingleTap {
    if (shiftButtonState == LOWER) {
        [self switchAllKeys:true];
        shiftButtonState = SHIFT;
        [shiftButton setBackgroundImage:[UIImage imageNamed:@"Shift.png"] forState:UIControlStateNormal];
    } else {
        [self switchAllKeys:false];
        shiftButtonState = LOWER;
        [shiftButton setBackgroundImage:[UIImage imageNamed:@"Lower.png"] forState:UIControlStateNormal];
    }
}

- (void)shiftDoubleTap {
    [self switchAllKeys:true];
    shiftButtonState = UPPER;
    [shiftButton setBackgroundImage:[UIImage imageNamed:@"Upper.png"] forState:UIControlStateNormal];
}

- (void)switchAllKeys:(bool)uppercase {
    for(UIView *v in [keyPad subviews]) {
        if ([v isKindOfClass:[UIButton class]]) {
            NSString *label = [(UIButton *)v currentTitle];
            if (label.length == 1) {
                if (uppercase) {
                    [(UIButton *)v setTitle:[label uppercaseString] forState:UIControlStateNormal];
                } else {
                    [(UIButton *)v setTitle:[label lowercaseString] forState:UIControlStateNormal];
                }
            }
        }
    }
}

- (IBAction)backspace:(id)sender {
    NSArray *tokens = [self.textDocumentProxy.documentContextBeforeInput componentsSeparatedByString:@" "];
    unsigned long num_tokens = [tokens count];
    if (num_tokens > 1 && [[tokens objectAtIndex:(num_tokens - 2)] length] > 0 && [[tokens lastObject] length] == 0) {
        usePrevStateRNN = false;
        [self.textDocumentProxy deleteBackward];
        [self newPredictions];
    } else {
        [self.textDocumentProxy deleteBackward];
        [self updatePredictions];
    }
}

- (IBAction)switchPad:(id)sender {
    if (padState == LETTERS) {
        [self.view bringSubviewToFront:numberPad];
        [padButton setTitle:@"ABC" forState:UIControlStateNormal];
        padState = NUMBERS;
    } else {
        [self.view bringSubviewToFront:keyPad];
        [padButton setTitle:@"123" forState:UIControlStateNormal];
        if (padState == SYMBOLS) {
            [self switchSymbols:nil];
        }
        padState = LETTERS;
    }
}

- (IBAction)switchSymbols:(id)sender {
    if (padState == NUMBERS) {
        for(UIView *v in [numberPad subviews]) {
            if ([v isKindOfClass:[UIButton class]]) {
                NSString *currentTitle = [(UIButton *)v currentTitle];
                [(UIButton *)v setTitle:[symbolPairs1 objectForKey:currentTitle] forState:UIControlStateNormal];
            }
        }
        [symbolButton setTitle:@"123" forState:UIControlStateNormal];
        padState = SYMBOLS;
    } else if (padState == SYMBOLS) {
        for(UIView *v in [numberPad subviews]) {
            if ([v isKindOfClass:[UIButton class]]) {
                NSString *currentTitle = [(UIButton *)v currentTitle];
                [(UIButton *)v setTitle:[symbolPairs2 objectForKey:currentTitle] forState:UIControlStateNormal];
            }
        }
        [symbolButton setTitle:@"#+=" forState:UIControlStateNormal];
        padState = NUMBERS;
    }
}

- (IBAction)nextKeyboard:(id)sender {
    [self advanceToNextInputMode];
}

- (void)newPredictions {
    std::list<std::string> seq;
    
    NSArray *tokens = [self.textDocumentProxy.documentContextBeforeInput componentsSeparatedByString:@" "];
    unsigned long numTokens = [tokens count];
    
    if (usePrevStateRNN) {
        seq.push_back(std::string([[[tokens objectAtIndex:(numTokens - 2)] lowercaseString] UTF8String]));
    } else {
        seq.push_back("<s>");
        for (id s in tokens) {
            seq.push_back(std::string([[s lowercaseString] UTF8String]));
        }
    }
    
    rnn->ProbAllFollowing(seq, charTrie, usePrevStateRNN);
    std::list<std::pair<std::string, double>> top3 = charTrie->GetMaxKWithPrefix([[tokens lastObject] UTF8String], 3);
    [self setPredictionsWithTop3:&top3];
    
    if (!usePrevStateRNN) {
        usePrevStateRNN = true;
    }
}

- (void)updatePredictions {
    NSArray *tokens = [self.textDocumentProxy.documentContextBeforeInput componentsSeparatedByString:@" "];
    std::string prefix = "";
    if ([tokens count] > 0) {
        prefix = [[[tokens lastObject] lowercaseString] UTF8String];
    }
    std::list<std::pair<std::string, double>> top3 = charTrie->GetMaxKWithPrefix(prefix, 3);
    [self setPredictionsWithTop3:&top3];
}

- (void)setPredictionsWithTop3:(std::list<std::pair<std::string, double>> *)top3 {
    // Check if we're currently predicting words following a full stop.
    bool camelCase = false;
    if (shiftButtonState != UPPER) {
        NSArray *tokens = [self.textDocumentProxy.documentContextBeforeInput componentsSeparatedByString:@" "];
        if ([tokens count] <= 1) {
            camelCase = true;
        } else {
            long i = [tokens count] - 2;
            NSString *tmp = @"";
            while (i >= 0 && [tmp length] == 0) {
                tmp = [tokens objectAtIndex:i];
                if ([tmp length] > 0) {
                    if ([tmp characterAtIndex:[tmp length] - 1] == '.') {
                        camelCase = true;
                    }
                }
                i--;
            }
        }
    }

    for (id button in predictionButtons) {
        if (top3->size() == 0) return;
        NSString *title = [NSString stringWithCString:top3->front().first.c_str() encoding:[NSString defaultCStringEncoding]];
        if (shiftButtonState == UPPER) {
            title = [title uppercaseString];
        } else if (camelCase) {
            title = [title capitalizedString];
        }
        [(UIButton *)button setTitle:title forState:UIControlStateNormal];
        top3->pop_front();
    }
}

@end
