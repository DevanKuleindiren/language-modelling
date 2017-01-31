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
    
    caps_on = true;
    usePrevStateRNN = false;
    
    predictionButtons = [[NSArray alloc] initWithObjects:firstPrediction, secondPrediction, thirdPrediction, nil];
    
    // Load RNN.
    NSString* model_path = [[NSBundle mainBundle] pathForResource:@"graph" ofType:@"pb"];
    NSRange range = [model_path rangeOfString: @"/" options: NSBackwardsSearch];
    rnn = new RNN(std::string([[model_path substringToIndex: range.location] UTF8String]));
    
    std::list<std::pair<std::string, double>> probs;
    rnn->ProbAllFollowing(std::list<std::string>({"<s>"}), probs);
    charTrie = new CharTrie();
    for (std::list<std::pair<std::string, double>>::const_iterator it = probs.begin(); it != probs.end(); ++it) {
        if (!(it->first.compare("<unk>") == 0 ||
              it->first.compare("N") == 0 ||
              it->first.compare("<s>") == 0)) {
            charTrie->Insert(it->first, it->second);
        }
    }
    std::list<std::pair<std::string, double>> top3 = charTrie->GetMaxKWithPrefix("", 3);
    [self setPredictionsWithTop3:&top3];
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
    unsigned long num_tokens = [tokens count];
    if (num_tokens > 1 && [[tokens objectAtIndex:(num_tokens - 2)] length] > 0 && [[tokens lastObject] length] == 0) {
        [self newPredictions];
    } else {
        [self updatePredictions];
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

- (IBAction)caps:(id)sender {
    caps_on = !caps_on;
    for(UIView *v in [self.view subviews]) {
        if ([v isKindOfClass:[UIButton class]]) {
            NSString *label = [(UIButton *)v currentTitle];
            if (label.length == 1) {
                if (caps_on) {
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

- (IBAction)nextKeyboard:(id)sender {
    [self advanceToNextInputMode];
}

- (void)newPredictions {
    std::list<std::pair<std::string, double>> probs;
    std::list<std::string> seq;
    
    NSArray *tokens = [self.textDocumentProxy.documentContextBeforeInput componentsSeparatedByString:@" "];
    unsigned long numTokens = [tokens count];
    
    if (numTokens <= 1) {
        seq.push_back("<s>");
    } else {
        if (usePrevStateRNN) {
            seq.push_back(std::string([[tokens objectAtIndex:(numTokens - 2)] UTF8String]));
        } else {
            for (id s in tokens) {
                seq.push_back(std::string([s UTF8String]));
            }
        }
    }
    
    rnn->ProbAllFollowing(seq, probs, usePrevStateRNN);
    for (std::list<std::pair<std::string, double>>::const_iterator it = probs.begin(); it != probs.end(); ++it) {
        charTrie->Update(it->first, it->second);
    }
    
    std::list<std::pair<std::string, double>> top3 = charTrie->GetMaxKWithPrefix([[tokens lastObject] UTF8String], 3);
    [self setPredictionsWithTop3:&top3];
    
    if (!usePrevStateRNN) {
        usePrevStateRNN = true;
    }
}

- (void)updatePredictions {
    NSArray *tokens = [self.textDocumentProxy.documentContextBeforeInput componentsSeparatedByString:@" "];
    std::list<std::pair<std::string, double>> top3 = charTrie->GetMaxKWithPrefix([[tokens lastObject] UTF8String], 3);
    [self setPredictionsWithTop3:&top3];
}

- (void)setPredictionsWithTop3:(std::list<std::pair<std::string, double>> *)top3 {
    for (id button in predictionButtons) {
        if (top3->size() == 0) return;
        [(UIButton *)button setTitle:[NSString stringWithCString:top3->front().first.c_str() encoding:[NSString defaultCStringEncoding]] forState:UIControlStateNormal];
        top3->pop_front();
    }
}

@end
