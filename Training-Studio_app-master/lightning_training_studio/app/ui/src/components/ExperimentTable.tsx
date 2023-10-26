import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import PlayCircleIcon from '@mui/icons-material/PlayCircle';
import StopCircle from '@mui/icons-material/StopCircle';
import useShowHelpPageState, { HelpPageState } from 'hooks/useShowHelpPageState';
import { Box, Table, Typography } from 'lightning-ui/src/design-system/components';
import { StatusEnum } from 'lightning-ui/src/shared/components/Status';
import { AppClient, ExperimentConfig, SweepConfig, TensorboardConfig } from '../generated';
import useClientDataState from '../hooks/useClientDataState';
import { formatDurationFrom, formatDurationStartEnd, getAppId } from '../utilities';
import BorderLinearProgress from './BorderLinearProgress';
import MoreMenu from './MoreMenu';
import UserGuide, { UserGuideBody, UserGuideComment } from './UserGuide';

const appClient = new AppClient({
  BASE:
    window.location != window.parent.location
      ? document.referrer.replace(/\/$/, '').replace('/view/undefined', '')
      : document.location.href.replace(/\/$/, '').replace('/view/undefined', ''),
});

const statusToEnum = {
  not_started: StatusEnum.NOT_STARTED,
  pending: StatusEnum.PENDING,
  running: StatusEnum.RUNNING,
  pruned: StatusEnum.DELETED,
  succeeded: StatusEnum.SUCCEEDED,
  failed: StatusEnum.FAILED,
  stopped: StatusEnum.STOPPED,
} as { [k: string]: StatusEnum };

const ComputeToMachines = {
  'cpu': '1 CPU',
  'cpu-small': '2 CPU',
  'cpu-medium': '8 CPU',
  'gpu': '1 T4',
  'gpu-fast': '1 V100',
  'gpu-fast-multi': '4 V100',
} as { [k: string]: string };

const StageToColors = {
  succeeded: 'success',
  failed: 'error',
  pending: 'primary',
  running: 'primary',
} as { [k: string]: 'success' | 'error' | 'primary' };

function stopTensorboard(tensorboardConfig: TensorboardConfig) {
  appClient.appApi.stopTensorboardApiStopTensorboardPost({ sweep_id: tensorboardConfig.sweep_id });
}

function runTensorboard(tensorboardConfig?: TensorboardConfig) {
  if (tensorboardConfig) {
    appClient.appApi.runTensorboardApiRunTensorboardPost({
      id: tensorboardConfig.id,
      sweep_id: tensorboardConfig.sweep_id,
      shared_folder: tensorboardConfig.shared_folder,
      stage: StatusEnum.RUNNING.toLowerCase(),
      desired_stage: StatusEnum.RUNNING.toLowerCase(),
      url: undefined,
    });
  }
}

function toCompute(sweep: SweepConfig) {
  if (sweep.cloud_compute) {
    if (sweep.num_nodes && sweep.num_nodes > 1) {
      return `${sweep.num_nodes} nodes x ${ComputeToMachines[sweep.cloud_compute]}`;
    }
    return ComputeToMachines[sweep.cloud_compute];
  }
  return '';
}

function toProgress(experiment: ExperimentConfig) {
  if (experiment.stage == 'pending' && !experiment.progress) {
    return (
      <Box sx={{ minWidth: 35 }}>
        <Typography variant="caption" display="block">{`Pending...`}</Typography>
      </Box>
    );
  } else if (experiment.stage == 'stopped') {
    return (
      <Box sx={{ minWidth: 35 }}>
        <Typography variant="caption" display="block">{`Stopped`}</Typography>
      </Box>
    );
  } else if (experiment.stage == 'failed') {
    return (
      <Box sx={{ minWidth: 35 }}>
        <Typography variant="caption" display="block">{`Failed`}</Typography>
      </Box>
    );
  } else if (experiment.stage == 'pruned') {
    return (
      <Box sx={{ minWidth: 35 }}>
        <Typography variant="caption" display="block">{`Pruned`}</Typography>
      </Box>
    );
  } else if (experiment.stage == 'succeeded') {
    return (
      <Box sx={{ minWidth: 35 }}>
        <Typography variant="caption" display="block">{`Succeeded`}</Typography>
      </Box>
    );
  } else {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center' }}>
        <Box sx={{ width: '100%', mr: 1 }}>
          <BorderLinearProgress
            variant={experiment.progress == 0 ? undefined : 'determinate'}
            color={StageToColors[experiment.stage]}
            value={experiment.progress}
          />
        </Box>
        {experiment.progress ? (
          <Box sx={{ minWidth: 35 }}>
            <Typography variant="caption" display="block">{`${experiment.progress}%`}</Typography>
          </Box>
        ) : (
          <Box></Box>
        )}
      </Box>
    );
  }
}

function runtimeTime(experiment: ExperimentConfig) {
  if (experiment.end_time && experiment.progress) {
    return formatDurationStartEnd(Number(experiment.end_time), Number(experiment.start_time));
  }
  return experiment.start_time ? String(formatDurationFrom(Number(experiment.start_time))) : '';
}

function getTime() {
  return Number(new Date().getTime() / 1000);
}

function timeLeft(experiment: ExperimentConfig) {
  if (experiment.end_time || experiment.stage == 'stopped' || experiment.stage == 'failed') {
    return '';
  }
  if (experiment.progress) {
    const nowTime = getTime();
    const estimatedEnd =
      Number(experiment.start_time) + (100 * (nowTime - Number(experiment.start_time))) / experiment.progress;
    return experiment.start_time ? String(formatDurationStartEnd(estimatedEnd, nowTime)) : '';
  }
}

function toArgs(
  script_args?: Array<string>,
  params?: Record<string, number | string | Array<number> | Array<string>>,
) {
  var arg = '';
  if (script_args && script_args.length > 0) {
    arg = arg + script_args.join(' ') + ' ';
  }
  if (params) {
    for (var p in params) {
      arg = arg + `${p}=${params[p]} `;
    }
  }
  return arg;
}

const handleClick = (url?: string) => {
  if (url) {
    window.open(url, '_blank')?.focus();
  }
};

function createMenuItems(logger_url?: string, tensorboardConfig?: TensorboardConfig) {
  var items = [];
  const url = tensorboardConfig ? tensorboardConfig.url : logger_url;
  const status = tensorboardConfig?.stage ? statusToEnum[tensorboardConfig.stage] : StatusEnum.NOT_STARTED;

  items.push({
    label: 'Open Tensorboard',
    icon: <OpenInNewIcon sx={{ fontSize: 20 }} />,
    onClick: () => handleClick(url),
    disabled: status != StatusEnum.RUNNING,
  });

  if (tensorboardConfig) {
    const status = tensorboardConfig?.stage ? statusToEnum[tensorboardConfig.stage] : StatusEnum.NOT_STARTED;

    if (status == StatusEnum.RUNNING) {
      items.push({
        label: 'Stop Tensorboard',
        icon: <StopCircle sx={{ fontSize: 20 }} />,
        onClick: () => stopTensorboard(tensorboardConfig),
      });
    } else if (status == StatusEnum.STOPPED) {
      items.push({
        label: 'Start Tensorboard',
        icon: <PlayCircleIcon sx={{ fontSize: 20 }} />,
        onClick: () => runTensorboard(tensorboardConfig),
      });
    }
  }

  return items;
}

export function Experiments() {
  const { showHelpPage, setShowHelpPage } = useShowHelpPageState();
  const tensorboards = useClientDataState('tensorboards') as TensorboardConfig[];
  const sweeps = useClientDataState('sweeps') as SweepConfig[];

  const appId = getAppId();

  if (sweeps.length == 0) {
    setShowHelpPage(HelpPageState.forced);
  } else if (showHelpPage == HelpPageState.forced) {
    setShowHelpPage(HelpPageState.notShown);
  }

  if (showHelpPage == HelpPageState.forced || showHelpPage == HelpPageState.shown) {
    return (
      <UserGuide
        title="Want to start a hyper-parameter sweep?"
        subtitle="Use the commands below in your local terminal on your own computer.">
        <UserGuideComment>Create a new Conda Env</UserGuideComment>
        <UserGuideBody>{`conda create --yes --name training-studio python=3.9`}</UserGuideBody>
        <UserGuideComment>Activate the conda env</UserGuideComment>
        <UserGuideBody>{`conda activate training-studio`}</UserGuideBody>
        <UserGuideComment>Install Lightning</UserGuideComment>
        <UserGuideBody>{`pip install lightning`}</UserGuideBody>
        <UserGuideComment>Connect to the app</UserGuideComment>
        <UserGuideBody>{`lightning connect ${appId}`}</UserGuideBody>
        <UserGuideComment>Create a new folder</UserGuideComment>
        <UserGuideBody>{`mkdir new_folder && cd new_folder`}</UserGuideBody>
        <UserGuideComment>Download example script</UserGuideComment>
        <UserGuideBody>
          {
            'curl -o train.py https://raw.githubusercontent.com/Lightning-AI/lightning-hpo/master/sweep_examples/scripts/train.py'
          }
        </UserGuideBody>
        <UserGuideComment>Run a sweep</UserGuideComment>
        <UserGuideBody>
          lightning run sweep train.py --model.lr "[0.001, 0.01]" --data.batch "[32, 64]" --algorithm="grid_search"
          --cloud_compute="cpu-medium"
        </UserGuideBody>
      </UserGuide>
    );
  }

  const experimentHeader = [
    'Progress',
    'Runtime',
    //'Time Left',
    'Sweep Name',
    'Name',
    'Monitor',
    'Best Score',
    'Script Arguments',
    'Data',
    'Trainable Parameters',
    'Compute',
    'More',
  ];

  const tensorboardIdsToStatuses = Object.fromEntries(
    tensorboards.map(e => {
      return [e.sweep_id, e];
    }),
  );

  var rows = sweeps.map(sweep => {
    const tensorboardConfig = tensorboardIdsToStatuses[sweep.sweep_id];

    const data = Object.entries(sweep.data)
      .map(entry => {
        if (entry[1]) {
          return `${entry[0]}:${entry[1]}`;
        }
        return entry[0];
      })
      .join(', ');

    return Object.entries(sweep.experiments).map(entry => [
      toProgress(entry[1]),
      runtimeTime(entry[1]),
      //timeLeft(entry[1]),
      sweep.sweep_id,
      entry[1].name,
      String(entry[1].monitor),
      entry[1].best_model_score && String(entry[1].best_model_score.toPrecision(4)),
      toArgs(sweep.script_args, entry[1].params),
      data,
      String(entry[1].total_parameters),
      toCompute(sweep),
      <MoreMenu id={entry[1].name} items={createMenuItems(sweep.logger_url, tensorboardConfig)} />,
    ]);
  });

  const flatArray = rows.flat().map((row: any[]) =>
    row.map((entry: any) => {
      if (!entry || entry == 'null' || entry == 'None') {
        return '-';
      }
      return entry;
    }),
  );

  return <Table header={experimentHeader} rows={flatArray} />;
}
