// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React from 'react';
import Icon from '@ant-design/icons';

import { MultiSelectIcon } from 'icons';
import { Canvas } from 'cvat-canvas-wrapper';
import { ActiveControl } from 'reducers';
import CVATTooltip from 'components/common/cvat-tooltip';

export interface Props {
    updateActiveControl(activeControl: ActiveControl): void;
    canvasInstance: Canvas;
    disabled?: boolean;
    activeControl: ActiveControl;
}

function MultiSelectControl(props: Props): JSX.Element {
    const {
        updateActiveControl,
        canvasInstance,
        activeControl,
        disabled,
    } = props;

    const isActive = activeControl === ActiveControl.MULTISELECT;

    const onClick = (): void => {
        if (isActive) {
            canvasInstance.multiselect(false);
            updateActiveControl(ActiveControl.CURSOR);
        } else {
            canvasInstance.cancel();
            canvasInstance.multiselect(true);
            updateActiveControl(ActiveControl.MULTISELECT);
        }
    };

    return disabled ? (
        <Icon className='cvat-multiselect-control cvat-disabled-canvas-control' component={MultiSelectIcon} />
    ) : (
        <CVATTooltip title='Multi-select masks to change their label' placement='right'>
            <Icon
                className={`cvat-multiselect-control${isActive ? ' cvat-active-canvas-control' : ''}`}
                component={MultiSelectIcon}
                onClick={onClick}
            />
        </CVATTooltip>
    );
}

export default React.memo(MultiSelectControl);
