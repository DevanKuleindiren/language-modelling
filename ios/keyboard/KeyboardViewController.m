//
//  KeyboardViewController.m
//  keyboard
//
//  Created by Devan Kuleindiren on 27/01/2017.
//  Copyright © 2017 Google. All rights reserved.
//

#import "KeyboardViewController.h"

@interface KeyboardViewController ()
@property (nonatomic, strong) UIButton *nextKeyboardButton;
@end

@implementation KeyboardViewController

- (void)updateViewConstraints {
    [super updateViewConstraints];
    
    // Add custom view sizing constraints here
}

- (void)viewDidLoad {
    [super viewDidLoad];
    
    caps_on = true;
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
    
    UIColor *textColor = nil;
    if (self.textDocumentProxy.keyboardAppearance == UIKeyboardAppearanceDark) {
        textColor = [UIColor whiteColor];
    } else {
        textColor = [UIColor blackColor];
    }
    [self.nextKeyboardButton setTitleColor:textColor forState:UIControlStateNormal];
}

- (void)insertString:(NSString *)s {
    [self.textDocumentProxy insertText:s];
}

-(IBAction)keyPress:(id)sender {
    if (![sender isKindOfClass:[UIButton class]]) {
        return;
    }
    [self insertString:[(UIButton *)sender currentTitle]];
}

- (IBAction)newLine:(id)sender {
    [self insertString:@"\n"];
}

- (IBAction)caps:(id)sender {
    caps_on = !caps_on;
    for(UIView *v in [self.view subviews]) {
        if ([v isKindOfClass:[UIButton class]]) {
            NSString *label = [(UIButton *)v currentTitle];
            if (label.length == 1) {
                [UIView setAnimationsEnabled:NO];
                if (caps_on) {
                    [(UIButton *)v setTitle:[label uppercaseString] forState:UIControlStateNormal];
                } else {
                    [(UIButton *)v setTitle:[label lowercaseString] forState:UIControlStateNormal];
                }
                [UIView setAnimationsEnabled:YES];
            }
        }
    }
}

- (IBAction)backspace:(id)sender {
    [self.textDocumentProxy deleteBackward];
}

- (IBAction)nextKeyboard:(id)sender {
    [self advanceToNextInputMode];
}

@end
