import { Divider, Grid, Typography } from '@mui/material';
import MuiTab from '@mui/material/Tab';
import MuiTabs from '@mui/material/Tabs';
import useShowHelpPageState, { HelpPageState } from 'hooks/useShowHelpPageState';
import { Box, IconButton, Stack, SxProps, Theme } from 'lightning-ui/src/design-system/components';
import TabContent from 'lightning-ui/src/design-system/components/tabs/TabContent';
import TabPanel from 'lightning-ui/src/design-system/components/tabs/TabPanel';
import React, { ReactNode } from 'react';

export type TabItem = {
  title: string;
  content: ReactNode;
};

export type TabsProps = {
  selectedTab: number;
  onChange: (selectedTab: number) => void;
  tabItems: TabItem[];
  variant?: 'text' | 'outlined';
  sxTabs?: SxProps<Theme>;
  sxContent?: SxProps<Theme>;
};

const Tabs = (props: TabsProps) => {
  const { showHelpPage, setShowHelpPage } = useShowHelpPageState();

  return (
    <Box sx={{ overflowX: 'hidden' }}>
      <Stack
        direction="row"
        spacing={1}
        sx={{ paddingX: '14px', paddingY: '9px', position: 'absolute', top: 1, right: 0, zIndex: 1000 }}>
        {showHelpPage != HelpPageState.forced ? (
          <IconButton
            sx={{
              'backgroundColor': 'grey.20',
              '&:hover': {
                backgroundColor: 'grey.20',
              },
              'height': '50px',
              'width': '50px',
            }}
            aria-label="commands"
            onClick={() => {
              setShowHelpPage(showHelpPage == HelpPageState.shown ? HelpPageState.notShown : HelpPageState.shown);
            }}>
            {showHelpPage == HelpPageState.shown ? (
              <Box sx={{ minWidth: 35 }}>
                <Typography sx={{ fontSize: 12 }} color="grey.70">
                  <b>CLI Help</b>
                </Typography>
              </Box>
            ) : (
              <Box sx={{ minWidth: 35 }}>
                <Typography sx={{ fontSize: 12 }} color="primary">
                  <b>CLI Help</b>
                </Typography>
              </Box>
            )}
          </IconButton>
        ) : (
          <></>
        )}
      </Stack>
      <Grid container spacing={1}>
        <Grid item xs={12} sm="auto">
          <Box sx={{ marginX: '20px', marginY: '20px' }}>
            <Typography variant="h6">Training Studio</Typography>
          </Box>
        </Grid>
        <Grid item xs={12} sm>
          <MuiTabs
            value={props.selectedTab}
            onChange={(e, value) => props.onChange(value)}
            variant={'scrollable'}
            sx={props.sxTabs}>
            {props.tabItems.map((tabItem: any, index) => (
              // @ts-ignore
              <MuiTab key={tabItem.title} label={tabItem.title} variant={props.variant} />
            ))}
          </MuiTabs>
        </Grid>
      </Grid>
      <Divider />
      <Box paddingY="0px" paddingX="0px" sx={props.sxContent}>
        {props.tabItems.map((tabItem: any, index) => (
          <TabPanel key={index.toString()} value={props.selectedTab} index={index}>
            <TabContent>{tabItem.content}</TabContent>
          </TabPanel>
        ))}
      </Box>
    </Box>
  );
};

export default Tabs;
